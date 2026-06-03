# Cosmos Gremlin Graph Presenter

A Streamlit app for browsing an Azure Cosmos DB Gremlin graph.

## Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Connection String

The app accepts common Cosmos connection-string keys:

```text
AccountEndpoint=https://<account>.documents.azure.com:443/;AccountKey=<key>;Database=<database>
```

You can also use a Gremlin endpoint:

```text
AccountEndpoint=wss://<account>.gremlin.cosmos.azure.com:443/;AccountKey=<key>;Database=<database>
```

Enter the connection string in the first text box and the graph/container name in the second text box. If `Database` is not present in the connection string, expand **Advanced options** and enter it there.
