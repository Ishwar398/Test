import json
import os
import sys
import time
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader

try:
    from streamlit.runtime import exists as _streamlit_runtime_exists
except ImportError:  # pragma: no cover - older Streamlit versions
    _streamlit_runtime_exists = None

if _streamlit_runtime_exists is not None and not _streamlit_runtime_exists():
    sys.stderr.write(
        "\nThis is a Streamlit app — launch it with:\n\n"
        "    streamlit run app.py\n\n"
        "Running it via `python app.py` will not work because Streamlit's "
        "session state and widgets require its own runtime.\n"
    )
    sys.exit(1)

load_dotenv()

st.set_page_config(page_title="RAG API Tester", layout="wide")
st.title("RAG API Tester")
st.caption("Generate questions from a PDF and fire them at your RAG endpoint.")


def _init_state() -> None:
    defaults = {
        "pdf_text": "",
        "pdf_name": "",
        "generated_questions": [],
        "body_template": '{\n  "query": "{{question}}"\n}',
        "extra_headers": "{}",
        "single_response": None,
        "batch_results": [],
        "answer_path": "answer",
        "context_path": "",
        "expected_answers": {},
        "evaluation_results": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n".join(parts).strip()


def build_azure_client(endpoint: str, api_key: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def generate_questions(
    client: AzureOpenAI,
    deployment: str,
    text: str,
    num_questions: int,
    style: str,
) -> list[str]:
    snippet = text[:12000]
    style_hint = {
        "Factual": "Ask concrete, fact-seeking questions that have clear answers in the text.",
        "Analytical": "Ask questions that require reasoning, comparison, or synthesis across the text.",
        "Mixed": "Mix factual and analytical questions for broad coverage.",
    }[style]
    system = (
        "You generate evaluation questions for a Retrieval-Augmented Generation (RAG) system. "
        "Return ONLY a JSON array of strings — no prose, no markdown fences."
    )
    user = (
        f"{style_hint}\n"
        f"Generate exactly {num_questions} diverse questions grounded in the document below. "
        f"Each question must be answerable from the document. "
        f"Avoid duplicates and yes/no questions.\n\n"
        f"--- DOCUMENT START ---\n{snippet}\n--- DOCUMENT END ---"
    )
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    questions = json.loads(raw)
    return [str(q).strip() for q in questions if str(q).strip()]


def substitute_placeholder(obj: Any, placeholder: str, value: str) -> Any:
    if isinstance(obj, dict):
        return {k: substitute_placeholder(v, placeholder, value) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_placeholder(item, placeholder, value) for item in obj]
    if isinstance(obj, str):
        return obj.replace(placeholder, value)
    return obj


def send_request(
    url: str,
    body_template: str,
    extra_headers_json: str,
    bearer_token: str,
    question: str,
    timeout: int,
) -> dict:
    body_obj = json.loads(body_template)
    body_obj = substitute_placeholder(body_obj, "{{question}}", question)

    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    if extra_headers_json.strip():
        headers.update(json.loads(extra_headers_json))

    start = time.perf_counter()
    response = requests.post(url, json=body_obj, headers=headers, timeout=timeout)
    elapsed = time.perf_counter() - start

    try:
        parsed = response.json()
    except ValueError:
        parsed = None

    return {
        "status_code": response.status_code,
        "elapsed": elapsed,
        "headers": dict(response.headers),
        "json": parsed,
        "text": response.text,
        "request_body": body_obj,
    }


def extract_by_path(obj: Any, path: str) -> Any:
    """Walk a dot-separated path through nested dicts/lists.

    Supports list indices as numeric segments, e.g. 'choices.0.message.content'.
    Returns None if any segment is missing.
    """
    if not path or obj is None:
        return obj
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            if not part.lstrip("-").isdigit():
                return None
            idx = int(part)
            if -len(cur) <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _coerce_to_text(value: Any, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[truncated]"
    return text


def _judge(
    client: AzureOpenAI,
    deployment: str,
    system: str,
    user: str,
) -> dict:
    """Call the model and parse a JSON response of shape {score, reasoning}."""
    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    parsed = json.loads(raw)
    score = parsed.get("score")
    if isinstance(score, str):
        try:
            score = int(score)
        except ValueError:
            score = None
    return {
        "score": score,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def judge_faithfulness(
    client: AzureOpenAI, deployment: str, question: str, answer: str, source: str
) -> dict:
    system = (
        "You are a strict evaluator of factual faithfulness. "
        "Given a SOURCE document, a QUESTION, and an ANSWER, decide how well the ANSWER is "
        "grounded in the SOURCE. Penalise any claim not supported by the SOURCE. "
        "Return JSON exactly as {\"score\": <int 1-5>, \"reasoning\": <string>} where "
        "5 = every claim supported, 3 = partially supported, 1 = fabricated or contradicts source."
    )
    user = (
        f"SOURCE:\n{_coerce_to_text(source, 8000)}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{_coerce_to_text(answer)}"
    )
    return _judge(client, deployment, system, user)


def judge_relevance(
    client: AzureOpenAI, deployment: str, question: str, answer: str
) -> dict:
    system = (
        "You evaluate whether an ANSWER directly addresses the QUESTION. "
        "Ignore factual accuracy — only judge relevance and on-topic-ness. "
        "Return JSON exactly as {\"score\": <int 1-5>, \"reasoning\": <string>} where "
        "5 = directly answers, 3 = partially on-topic, 1 = unrelated or evasive."
    )
    user = f"QUESTION:\n{question}\n\nANSWER:\n{_coerce_to_text(answer)}"
    return _judge(client, deployment, system, user)


def judge_reference_match(
    client: AzureOpenAI,
    deployment: str,
    question: str,
    expected: str,
    actual: str,
) -> dict:
    system = (
        "You compare an ACTUAL answer to an EXPECTED reference answer for the same question. "
        "Score semantic equivalence, not exact wording. "
        "Return JSON exactly as {\"score\": <int 1-5>, \"reasoning\": <string>} where "
        "5 = same meaning, 3 = partial overlap, 1 = contradictory or unrelated."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED:\n{_coerce_to_text(expected)}\n\n"
        f"ACTUAL:\n{_coerce_to_text(actual)}"
    )
    return _judge(client, deployment, system, user)


def judge_context_precision(
    client: AzureOpenAI,
    deployment: str,
    question: str,
    context: str,
) -> dict:
    system = (
        "You evaluate retrieval quality. Given a QUESTION and a CONTEXT (the chunks the RAG "
        "system retrieved), score how relevant the CONTEXT is to answering the QUESTION. "
        "Return JSON exactly as {\"score\": <int 1-5>, \"reasoning\": <string>} where "
        "5 = highly relevant, fully sufficient, 3 = partially relevant, 1 = irrelevant noise."
    )
    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{_coerce_to_text(context, 8000)}"
    )
    return _judge(client, deployment, system, user)


def generate_expected_answer(
    client: AzureOpenAI, deployment: str, question: str, source: str
) -> str:
    system = (
        "You are a careful reader. Answer the QUESTION using ONLY information present in the SOURCE. "
        "If the SOURCE does not contain the answer, reply exactly: 'NOT IN SOURCE'. "
        "Keep the answer concise (1-3 sentences)."
    )
    user = f"SOURCE:\n{_coerce_to_text(source, 10000)}\n\nQUESTION:\n{question}"
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


with st.sidebar:
    st.header("Azure OpenAI")
    azure_endpoint = st.text_input(
        "Endpoint",
        value=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        placeholder="https://<resource>.openai.azure.com/",
    )
    azure_key = st.text_input(
        "API Key",
        type="password",
        value=os.getenv("AZURE_OPENAI_API_KEY", ""),
    )
    azure_deployment = st.text_input(
        "Deployment name",
        value=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    )
    azure_api_version = st.text_input(
        "API version",
        value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
    )

    st.divider()
    st.header("RAG API")
    api_url = st.text_input(
        "Endpoint URL",
        value=os.getenv("RAG_API_URL", ""),
        placeholder="https://api.example.com/rag/query",
    )
    bearer_token = st.text_input(
        "Bearer Token",
        type="password",
        value=os.getenv("RAG_BEARER_TOKEN", ""),
        help="Sent as 'Authorization: Bearer <token>' if provided.",
    )
    request_timeout = st.number_input(
        "Request timeout (s)", min_value=5, max_value=600, value=60, step=5
    )

tab_pdf, tab_single, tab_batch, tab_eval = st.tabs(
    ["1. PDF & Questions", "2. Single Request", "3. Batch Run", "4. Evaluate Quality"]
)

with tab_pdf:
    st.subheader("Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    col_a, col_b = st.columns([1, 1])
    with col_a:
        num_questions = st.slider("Number of questions", 1, 25, 5)
    with col_b:
        style = st.selectbox("Question style", ["Mixed", "Factual", "Analytical"])

    if pdf_file is not None and pdf_file.name != st.session_state.pdf_name:
        with st.spinner("Extracting text from PDF..."):
            st.session_state.pdf_text = extract_pdf_text(pdf_file)
            st.session_state.pdf_name = pdf_file.name
        st.success(
            f"Loaded '{pdf_file.name}' — {len(st.session_state.pdf_text):,} characters extracted."
        )

    if st.session_state.pdf_text:
        with st.expander("Preview extracted text"):
            preview = st.session_state.pdf_text[:5000]
            if len(st.session_state.pdf_text) > 5000:
                preview += "\n\n... [truncated]"
            st.text_area("Text", value=preview, height=240, disabled=True)

    if st.button("Generate Questions", type="primary", disabled=not st.session_state.pdf_text):
        if not (azure_endpoint and azure_key and azure_deployment and azure_api_version):
            st.error("Fill in all Azure OpenAI fields in the sidebar first.")
        else:
            try:
                with st.spinner("Asking Azure OpenAI to generate questions..."):
                    client = build_azure_client(azure_endpoint, azure_key, azure_api_version)
                    questions = generate_questions(
                        client,
                        azure_deployment,
                        st.session_state.pdf_text,
                        num_questions,
                        style,
                    )
                st.session_state.generated_questions = questions
                st.success(f"Generated {len(questions)} questions.")
            except json.JSONDecodeError:
                st.error("Model did not return valid JSON. Try again or pick a different deployment.")
            except Exception as exc:
                st.error(f"Question generation failed: {exc}")

    if st.session_state.generated_questions:
        st.subheader("Questions (editable)")
        st.caption("Edit any question inline; changes feed into the Single Request and Batch tabs.")
        updated = []
        for i, q in enumerate(st.session_state.generated_questions):
            edited = st.text_area(
                f"Q{i + 1}", value=q, key=f"question_edit_{i}", height=70
            )
            updated.append(edited)
        st.session_state.generated_questions = updated

        col_clear, col_add = st.columns(2)
        with col_clear:
            if st.button("Clear all questions"):
                st.session_state.generated_questions = []
                st.rerun()
        with col_add:
            if st.button("Add empty question"):
                st.session_state.generated_questions.append("")
                st.rerun()

with tab_single:
    st.subheader("Request body template")
    st.caption("Use `{{question}}` anywhere inside string values — it gets replaced before sending.")
    st.text_area("JSON body", height=180, key="body_template")

    st.subheader("Extra headers (optional)")
    st.text_area("JSON object", height=90, key="extra_headers")

    st.subheader("Question")
    source = st.radio(
        "Source",
        ["Pick from generated", "Type manually"],
        horizontal=True,
    )
    if source == "Pick from generated":
        if st.session_state.generated_questions:
            chosen = st.selectbox(
                "Generated questions",
                st.session_state.generated_questions,
                index=0,
            )
        else:
            chosen = ""
            st.info("No generated questions yet — generate some in Tab 1 or switch to manual.")
    else:
        chosen = st.text_area("Question", height=80, key="manual_question")

    send_disabled = not (api_url and chosen and st.session_state.body_template.strip())
    if st.button("Send Request", type="primary", disabled=send_disabled):
        try:
            with st.spinner("Calling RAG API..."):
                result = send_request(
                    api_url,
                    st.session_state.body_template,
                    st.session_state.extra_headers,
                    bearer_token,
                    chosen,
                    int(request_timeout),
                )
            st.session_state.single_response = {"question": chosen, **result}
        except json.JSONDecodeError as exc:
            st.error(f"Body or headers are not valid JSON: {exc}")
        except requests.RequestException as exc:
            st.error(f"HTTP error: {exc}")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

    if st.session_state.single_response:
        res = st.session_state.single_response
        st.divider()
        st.subheader("Response")
        m1, m2, m3 = st.columns(3)
        m1.metric("Status", res["status_code"])
        m2.metric("Latency", f"{res['elapsed']:.2f} s")
        m3.metric("Bytes", len(res["text"]))

        with st.expander("Request body sent", expanded=False):
            st.json(res["request_body"])
        with st.expander("Response headers", expanded=False):
            st.json(res["headers"])

        st.markdown("**Response body**")
        if res["json"] is not None:
            st.json(res["json"])
        else:
            st.code(res["text"] or "(empty)")

with tab_batch:
    st.subheader("Run all generated questions")
    if not st.session_state.generated_questions:
        st.info("Generate questions in Tab 1 first.")
    elif not api_url:
        st.warning("Set the RAG API endpoint URL in the sidebar.")
    else:
        st.write(f"Will send **{len(st.session_state.generated_questions)}** requests "
                 f"using the body template defined in Tab 2.")
        if st.button("Run batch", type="primary"):
            results = []
            progress = st.progress(0.0, text="Starting...")
            total = len(st.session_state.generated_questions)
            for i, q in enumerate(st.session_state.generated_questions, start=1):
                progress.progress(
                    (i - 1) / total, text=f"Sending {i}/{total}: {q[:60]}..."
                )
                try:
                    res = send_request(
                        api_url,
                        st.session_state.body_template,
                        st.session_state.extra_headers,
                        bearer_token,
                        q,
                        int(request_timeout),
                    )
                    results.append({"question": q, "ok": True, **res})
                except Exception as exc:
                    results.append({"question": q, "ok": False, "error": str(exc)})
            progress.progress(1.0, text="Done")
            st.session_state.batch_results = results

    if st.session_state.batch_results:
        st.divider()
        ok_count = sum(1 for r in st.session_state.batch_results if r.get("ok"))
        st.subheader(f"Results: {ok_count}/{len(st.session_state.batch_results)} succeeded")

        for i, r in enumerate(st.session_state.batch_results, start=1):
            status = (
                f"{r.get('status_code')} · {r.get('elapsed', 0):.2f}s"
                if r.get("ok")
                else "ERROR"
            )
            with st.expander(f"Q{i}: {r['question'][:80]}  —  {status}"):
                if not r.get("ok"):
                    st.error(r.get("error", "Unknown error"))
                    continue
                st.markdown("**Request body**")
                st.json(r["request_body"])
                st.markdown("**Response**")
                if r["json"] is not None:
                    st.json(r["json"])
                else:
                    st.code(r["text"] or "(empty)")

        export = [
            {
                "question": r["question"],
                "ok": r.get("ok", False),
                "status_code": r.get("status_code"),
                "elapsed_seconds": r.get("elapsed"),
                "response": r.get("json") if r.get("json") is not None else r.get("text"),
                "error": r.get("error"),
            }
            for r in st.session_state.batch_results
        ]
        st.download_button(
            "Download results as JSON",
            data=json.dumps(export, indent=2),
            file_name="rag_batch_results.json",
            mime="application/json",
        )

with tab_eval:
    st.subheader("Evaluate RAG output quality")
    st.caption(
        "Uses Azure OpenAI as judge to score the answers your API returned in Tab 3 "
        "against the question, the PDF source, and (optionally) an auto-generated reference answer."
    )

    successful = [r for r in st.session_state.batch_results if r.get("ok")]
    if not successful:
        st.info("Run a successful batch in Tab 3 first.")
    else:
        st.markdown(f"**{len(successful)}** successful responses available to evaluate.")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.text_input(
                "Answer path",
                key="answer_path",
                help="Dot-path inside the response JSON, e.g. `answer`, `data.text`, "
                "`choices.0.message.content`. Leave blank to use the entire response.",
            )
        with col_p2:
            st.text_input(
                "Context path (optional)",
                key="context_path",
                help="If your API returns retrieved context, point at it (e.g. `context` or "
                "`sources`). Used for Context Precision and overrides PDF for Faithfulness.",
            )

        st.markdown("**Metrics**")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            do_faithfulness = st.checkbox("Faithfulness", value=True)
        with m2:
            do_relevance = st.checkbox("Answer Relevance", value=True)
        with m3:
            do_match = st.checkbox("Reference Match", value=False)
        with m4:
            do_context = st.checkbox("Context Precision", value=False)

        if do_faithfulness and not st.session_state.context_path and not st.session_state.pdf_text:
            st.warning(
                "Faithfulness needs a source — either upload a PDF in Tab 1 or set a Context path."
            )

        if do_context and not st.session_state.context_path:
            st.warning("Context Precision requires a Context path.")

        with st.expander("Preview answer/context extraction"):
            sample = successful[0]
            extracted_answer = extract_by_path(sample.get("json"), st.session_state.answer_path)
            extracted_context = (
                extract_by_path(sample.get("json"), st.session_state.context_path)
                if st.session_state.context_path
                else None
            )
            st.markdown(f"Sample question: *{sample['question']}*")
            st.markdown("**Extracted answer**")
            st.code(_coerce_to_text(extracted_answer) or "(nothing at that path)")
            if st.session_state.context_path:
                st.markdown("**Extracted context**")
                st.code(_coerce_to_text(extracted_context) or "(nothing at that path)")

        if do_match:
            st.divider()
            st.markdown("**Reference answers**")
            have = sum(
                1
                for r in successful
                if st.session_state.expected_answers.get(r["question"])
            )
            st.caption(f"Have references for {have}/{len(successful)} questions.")
            col_gen, col_clr = st.columns(2)
            with col_gen:
                if st.button("Generate expected answers from PDF"):
                    if not st.session_state.pdf_text:
                        st.error("Upload a PDF in Tab 1 first.")
                    elif not (azure_endpoint and azure_key and azure_deployment):
                        st.error("Configure Azure OpenAI in the sidebar.")
                    else:
                        client = build_azure_client(
                            azure_endpoint, azure_key, azure_api_version
                        )
                        progress = st.progress(0.0, text="Generating references...")
                        total = len(successful)
                        for i, r in enumerate(successful, start=1):
                            q = r["question"]
                            progress.progress(
                                (i - 1) / total, text=f"Reference {i}/{total}"
                            )
                            try:
                                st.session_state.expected_answers[q] = (
                                    generate_expected_answer(
                                        client,
                                        azure_deployment,
                                        q,
                                        st.session_state.pdf_text,
                                    )
                                )
                            except Exception as exc:
                                st.session_state.expected_answers[q] = f"ERROR: {exc}"
                        progress.progress(1.0, text="Done")
                        st.success("References generated.")
            with col_clr:
                if st.button("Clear references"):
                    st.session_state.expected_answers = {}
                    st.rerun()

            if st.session_state.expected_answers:
                with st.expander("Inspect / edit references"):
                    for r in successful:
                        q = r["question"]
                        cur = st.session_state.expected_answers.get(q, "")
                        new_val = st.text_area(
                            q,
                            value=cur,
                            key=f"ref_{hash(q)}",
                            height=80,
                        )
                        st.session_state.expected_answers[q] = new_val

        st.divider()
        run_disabled = not any([do_faithfulness, do_relevance, do_match, do_context])
        if st.button("Run evaluation", type="primary", disabled=run_disabled):
            if not (azure_endpoint and azure_key and azure_deployment):
                st.error("Configure Azure OpenAI in the sidebar.")
            else:
                client = build_azure_client(azure_endpoint, azure_key, azure_api_version)
                eval_rows = []
                total = len(successful)
                progress = st.progress(0.0, text="Starting evaluation...")
                for i, r in enumerate(successful, start=1):
                    q = r["question"]
                    answer = extract_by_path(r.get("json"), st.session_state.answer_path)
                    if answer is None and not st.session_state.answer_path:
                        answer = r.get("json") if r.get("json") is not None else r.get("text")
                    context = (
                        extract_by_path(r.get("json"), st.session_state.context_path)
                        if st.session_state.context_path
                        else None
                    )
                    row = {
                        "question": q,
                        "answer": _coerce_to_text(answer, 4000),
                        "scores": {},
                        "reasoning": {},
                    }

                    progress.progress(
                        (i - 1) / total, text=f"Evaluating {i}/{total}: {q[:50]}..."
                    )

                    if do_faithfulness:
                        source = context if context else st.session_state.pdf_text
                        if source:
                            try:
                                res = judge_faithfulness(
                                    client,
                                    azure_deployment,
                                    q,
                                    row["answer"],
                                    _coerce_to_text(source, 8000),
                                )
                                row["scores"]["faithfulness"] = res["score"]
                                row["reasoning"]["faithfulness"] = res["reasoning"]
                            except Exception as exc:
                                row["reasoning"]["faithfulness"] = f"ERROR: {exc}"

                    if do_relevance:
                        try:
                            res = judge_relevance(
                                client, azure_deployment, q, row["answer"]
                            )
                            row["scores"]["relevance"] = res["score"]
                            row["reasoning"]["relevance"] = res["reasoning"]
                        except Exception as exc:
                            row["reasoning"]["relevance"] = f"ERROR: {exc}"

                    if do_match:
                        expected = st.session_state.expected_answers.get(q, "")
                        if expected and not expected.startswith("ERROR"):
                            try:
                                res = judge_reference_match(
                                    client,
                                    azure_deployment,
                                    q,
                                    expected,
                                    row["answer"],
                                )
                                row["scores"]["reference_match"] = res["score"]
                                row["reasoning"]["reference_match"] = res["reasoning"]
                                row["expected"] = expected
                            except Exception as exc:
                                row["reasoning"]["reference_match"] = f"ERROR: {exc}"
                        else:
                            row["reasoning"]["reference_match"] = (
                                "No reference answer — generate one above."
                            )

                    if do_context and context is not None:
                        try:
                            res = judge_context_precision(
                                client,
                                azure_deployment,
                                q,
                                _coerce_to_text(context, 8000),
                            )
                            row["scores"]["context_precision"] = res["score"]
                            row["reasoning"]["context_precision"] = res["reasoning"]
                        except Exception as exc:
                            row["reasoning"]["context_precision"] = f"ERROR: {exc}"

                    eval_rows.append(row)

                progress.progress(1.0, text="Done")
                st.session_state.evaluation_results = eval_rows
                st.success(f"Evaluated {len(eval_rows)} responses.")

    if st.session_state.evaluation_results:
        st.divider()
        st.subheader("Aggregate scores")

        def _avg(metric: str) -> float | None:
            vals = [
                row["scores"][metric]
                for row in st.session_state.evaluation_results
                if isinstance(row["scores"].get(metric), int)
            ]
            return sum(vals) / len(vals) if vals else None

        metric_labels = [
            ("faithfulness", "Faithfulness"),
            ("relevance", "Answer Relevance"),
            ("reference_match", "Reference Match"),
            ("context_precision", "Context Precision"),
        ]
        cols = st.columns(len(metric_labels))
        for col, (key, label) in zip(cols, metric_labels):
            avg = _avg(key)
            col.metric(label, f"{avg:.2f} / 5" if avg is not None else "—")

        st.subheader("Per-question detail")
        for i, row in enumerate(st.session_state.evaluation_results, start=1):
            score_summary = " · ".join(
                f"{label}: {row['scores'].get(key, '—')}"
                for key, label in metric_labels
                if key in row["scores"] or key in row["reasoning"]
            )
            with st.expander(f"Q{i}: {row['question'][:70]}  —  {score_summary}"):
                st.markdown("**Answer (extracted)**")
                st.code(row["answer"] or "(empty)")
                if "expected" in row:
                    st.markdown("**Expected**")
                    st.code(row["expected"])
                for key, label in metric_labels:
                    if key in row["scores"] or key in row["reasoning"]:
                        score = row["scores"].get(key, "—")
                        reasoning = row["reasoning"].get(key, "")
                        st.markdown(f"**{label}:** {score} / 5")
                        if reasoning:
                            st.caption(reasoning)

        st.download_button(
            "Download evaluation as JSON",
            data=json.dumps(st.session_state.evaluation_results, indent=2),
            file_name="rag_evaluation.json",
            mime="application/json",
        )
