from __future__ import annotations

import html
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import streamlit as st
import streamlit.components.v1 as components
from gremlin_python.driver import client, serializer
from gremlin_python.driver.protocol import GremlinServerError
from pyvis.network import Network


st.set_page_config(page_title="Cosmos Gremlin Graph Presenter", layout="wide")


@dataclass(frozen=True)
class CosmosGremlinConfig:
    endpoint: str
    username: str
    password: str


def parse_connection_string(connection_string: str) -> dict[str, str]:
    parts: dict[str, str] = {}
    for item in connection_string.strip().split(";"):
        if not item.strip() or "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts[key.strip().lower()] = value.strip()
    return parts


def normalize_gremlin_endpoint(raw_endpoint: str) -> str:
    endpoint = raw_endpoint.strip()
    if not endpoint:
        raise ValueError("Connection string must include AccountEndpoint.")

    parsed = urlparse(endpoint)
    host = parsed.hostname or endpoint

    if host.endswith(".documents.azure.com"):
        host = host.replace(".documents.azure.com", ".gremlin.cosmos.azure.com")
    elif not host.endswith(".gremlin.cosmos.azure.com"):
        host = host.rstrip("/")

    return f"wss://{host}:443/"


def build_config(connection_string: str, database: str, graph_name: str) -> CosmosGremlinConfig:
    parts = parse_connection_string(connection_string)
    endpoint = parts.get("accountendpoint") or parts.get("endpoint")
    key = (
        parts.get("accountkey")
        or parts.get("key")
        or parts.get("primarykey")
        or parts.get("password")
    )
    database_name = database or parts.get("database") or parts.get("dbname")

    if not endpoint:
        raise ValueError("Connection string must include AccountEndpoint.")
    if not key:
        raise ValueError("Connection string must include AccountKey.")
    if not database_name:
        raise ValueError("Database was not found in the connection string.")
    if not graph_name:
        raise ValueError("Graph name is required.")

    return CosmosGremlinConfig(
        endpoint=normalize_gremlin_endpoint(endpoint),
        username=f"/dbs/{database_name}/colls/{graph_name}",
        password=key,
    )


def submit_query(config: CosmosGremlinConfig, query: str, bindings: dict[str, Any] | None = None) -> list[Any]:
    gremlin_client = client.Client(
        config.endpoint,
        "g",
        username=config.username,
        password=config.password,
        message_serializer=serializer.GraphSONSerializersV2d0(),
    )
    try:
        callback = gremlin_client.submitAsync(query, bindings=bindings or {})
        if callback.result() is None:
            return []
        return callback.result().all().result()
    finally:
        gremlin_client.close()


def unwrap_graphson(value: Any) -> Any:
    if isinstance(value, dict):
        if "@value" in value and len(value) <= 2:
            return unwrap_graphson(value["@value"])
        return {str(k): unwrap_graphson(v) for k, v in value.items() if k != "@type"}
    if isinstance(value, list):
        return [unwrap_graphson(v) for v in value]
    return value


def flatten_value(value: Any) -> str:
    value = unwrap_graphson(value)
    if isinstance(value, list):
        return ", ".join(flatten_value(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}: {flatten_value(val)}" for key, val in value.items())
    return "" if value is None else str(value)


def normalize_vertex(row: dict[str, Any]) -> dict[str, Any]:
    row = unwrap_graphson(row)
    props = row.get("props") or {}
    return {
        "id": str(row.get("id", "")),
        "label": str(row.get("label", "vertex")),
        "props": props if isinstance(props, dict) else {},
    }


def normalize_edge(row: dict[str, Any]) -> dict[str, Any]:
    row = unwrap_graphson(row)
    props = row.get("props") or {}
    return {
        "id": str(row.get("id", "")),
        "label": str(row.get("label", "edge")),
        "outV": str(row.get("outV", "")),
        "inV": str(row.get("inV", "")),
        "props": props if isinstance(props, dict) else {},
    }


def fetch_graph(config: CosmosGremlinConfig, vertex_limit: int, edge_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vertices_query = """
g.V().limit(vertexLimit)
 .project('id','label','props')
 .by(id())
 .by(label())
 .by(valueMap())
"""
    edges_query = """
g.E().limit(edgeLimit)
 .project('id','label','outV','inV','props')
 .by(id())
 .by(label())
 .by(outV().id())
 .by(inV().id())
 .by(valueMap())
"""
    vertices = [
        normalize_vertex(row)
        for row in submit_query(config, vertices_query, {"vertexLimit": vertex_limit})
    ]
    edges = [
        normalize_edge(row)
        for row in submit_query(config, edges_query, {"edgeLimit": edge_limit})
    ]
    return vertices, edges


def fetch_neighborhood(config: CosmosGremlinConfig, node_id: str, depth: int, edge_limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    vertex_query = """
g.V(nodeId)
 .repeat(both().dedup())
 .emit()
 .times(depth)
 .dedup()
 .project('id','label','props')
 .by(id())
 .by(label())
 .by(valueMap())
"""
    edge_query = """
g.V(nodeId)
 .repeat(bothE().dedup().otherV().dedup())
 .emit()
 .times(depth)
 .bothE()
 .dedup()
 .limit(edgeLimit)
 .project('id','label','outV','inV','props')
 .by(id())
 .by(label())
 .by(outV().id())
 .by(inV().id())
 .by(valueMap())
"""
    vertices = [
        normalize_vertex(row)
        for row in submit_query(config, vertex_query, {"nodeId": node_id, "depth": depth})
    ]
    edges = [
        normalize_edge(row)
        for row in submit_query(
            config,
            edge_query,
            {"nodeId": node_id, "depth": depth, "edgeLimit": edge_limit},
        )
    ]
    vertex_ids = {vertex["id"] for vertex in vertices}
    scoped_edges = [
        edge for edge in edges if edge["outV"] in vertex_ids and edge["inV"] in vertex_ids
    ]
    return vertices, scoped_edges


def vertex_title(vertex: dict[str, Any]) -> str:
    props = vertex.get("props", {})
    lines = [f"id: {vertex['id']}", f"label: {vertex['label']}"]
    lines.extend(f"{key}: {flatten_value(value)}" for key, value in props.items())
    return html.escape("\n".join(lines))


def edge_title(edge: dict[str, Any]) -> str:
    props = edge.get("props", {})
    lines = [f"id: {edge['id']}", f"label: {edge['label']}"]
    lines.extend(f"{key}: {flatten_value(value)}" for key, value in props.items())
    return html.escape("\n".join(lines))


def display_name(vertex: dict[str, Any]) -> str:
    props = vertex.get("props", {})
    for key in ("name", "title", "label", "displayName"):
        if key in props:
            name = flatten_value(props[key])
            if name:
                return name
    return vertex["id"]


def render_graph(vertices: list[dict[str, Any]], edges: list[dict[str, Any]], height: int, selected_node: str | None = None) -> None:
    network = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2937",
    )
    network.barnes_hut(gravity=-2200, central_gravity=0.25, spring_length=140)

    vertex_ids = {vertex["id"] for vertex in vertices}
    for vertex in vertices:
        is_selected = selected_node and vertex["id"] == selected_node
        network.add_node(
            vertex["id"],
            label=display_name(vertex),
            title=vertex_title(vertex),
            group=vertex["label"],
            size=28 if is_selected else 16,
            color="#ef4444" if is_selected else None,
        )

    for edge in edges:
        if edge["outV"] not in vertex_ids or edge["inV"] not in vertex_ids:
            continue
        network.add_edge(
            edge["outV"],
            edge["inV"],
            label=edge["label"],
            title=edge_title(edge),
            arrows="to",
        )

    network.set_options(
        """
{
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true
  },
  "physics": {
    "stabilization": {
      "iterations": 180
    }
  },
  "edges": {
    "smooth": {
      "type": "dynamic"
    },
    "font": {
      "size": 10,
      "align": "middle"
    }
  },
  "nodes": {
    "shape": "dot",
    "font": {
      "size": 13
    }
  }
}
"""
    )

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as file:
        output_path = Path(file.name)
    network.write_html(str(output_path), notebook=False)
    components.html(output_path.read_text(encoding="utf-8"), height=height + 32, scrolling=True)


def search_vertices(vertices: list[dict[str, Any]], term: str) -> list[dict[str, Any]]:
    if not term.strip():
        return []
    pattern = re.compile(re.escape(term.strip()), re.IGNORECASE)
    matches = []
    for vertex in vertices:
        haystack = " ".join(
            [
                vertex["id"],
                vertex["label"],
                json.dumps(unwrap_graphson(vertex.get("props", {})), default=str),
            ]
        )
        if pattern.search(haystack):
            matches.append(vertex)
    return matches


st.title("Cosmos Gremlin Graph Presenter")

connection_string = st.text_input(
    "Cosmos Gremlin connection string",
    type="password",
    placeholder="AccountEndpoint=...;AccountKey=...;Database=...",
)
graph_name = st.text_input("Graph name", placeholder="Enter the graph/container name")

with st.expander("Advanced options"):
    database_name = st.text_input(
        "Database name",
        help="Only needed when the connection string does not include Database.",
    )
    vertex_limit = st.slider("Vertex limit", 25, 5000, 500, step=25)
    edge_limit = st.slider("Edge limit", 25, 10000, 1000, step=25)
    height = st.slider("Graph height", 420, 1000, 680, step=20)

load_graph = st.button("Load graph", type="primary")

if "vertices" not in st.session_state:
    st.session_state.vertices = []
if "edges" not in st.session_state:
    st.session_state.edges = []
if "config" not in st.session_state:
    st.session_state.config = None

if load_graph:
    try:
        config = build_config(connection_string, database_name, graph_name)
        with st.spinner("Loading graph from Cosmos DB..."):
            vertices, edges = fetch_graph(config, vertex_limit, edge_limit)
        st.session_state.config = config
        st.session_state.vertices = vertices
        st.session_state.edges = edges
        st.success(f"Loaded {len(vertices)} vertices and {len(edges)} edges.")
    except (ValueError, GremlinServerError, Exception) as exc:
        st.error(f"Could not load graph: {exc}")

vertices: list[dict[str, Any]] = st.session_state.vertices
edges: list[dict[str, Any]] = st.session_state.edges
config: CosmosGremlinConfig | None = st.session_state.config

if not vertices:
    st.info("Enter your Cosmos connection details, graph name, and load the graph.")
    st.stop()

left, right = st.columns([0.72, 0.28], gap="large")

with right:
    st.subheader("Search and Traverse")
    search_term = st.text_input("Search node", placeholder="Node id, label, or property")
    matches = search_vertices(vertices, search_term)
    if search_term:
        st.caption(f"{len(matches)} match(es) in the loaded graph.")

    options = {f"{display_name(vertex)}  [{vertex['label']}]  ({vertex['id']})": vertex["id"] for vertex in (matches or vertices)}
    selected_label = st.selectbox("Selected node", options=list(options.keys()))
    selected_node = options[selected_label]
    depth = st.slider("Traversal depth", 1, 4, 1)

    show_neighborhood = st.button("Show relationships", use_container_width=True)
    show_full_graph = st.button("Show entire loaded graph", use_container_width=True)

    selected_vertex = next((vertex for vertex in vertices if vertex["id"] == selected_node), None)
    if selected_vertex:
        st.write("Node properties")
        st.json(unwrap_graphson(selected_vertex), expanded=False)

with left:
    st.subheader("Graph")
    if show_neighborhood and config:
        try:
            with st.spinner("Loading node relationships..."):
                n_vertices, n_edges = fetch_neighborhood(config, selected_node, depth, edge_limit)
            st.caption(f"Neighborhood view: {len(n_vertices)} vertices, {len(n_edges)} edges.")
            render_graph(n_vertices, n_edges, height, selected_node)
        except (GremlinServerError, Exception) as exc:
            st.error(f"Could not load relationships: {exc}")
    else:
        if show_full_graph:
            st.caption(f"Full loaded graph: {len(vertices)} vertices, {len(edges)} edges.")
        render_graph(vertices, edges, height, selected_node)

st.divider()
st.caption("Tip: drag nodes, zoom with the mouse wheel, use the navigation controls, and hover nodes or edges to inspect details.")
