# Uyir_project_streamlit

Streamlit app for a Hashimoto's RAG assistant using ChromaDB and OpenAI via LangChain.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app_enhanced.py
```

Set `OPENAI_API_KEY` in your environment or via Streamlit secrets.

## Vector DB (Chroma)

This app expects a persisted Chroma DB at `chroma_db/`.

Because the DB can be large, do not commit it to Git. Instead provide a downloadable ZIP via an environment variable or Streamlit secret:

- `CHROMA_DB_URL`: HTTPS URL to a ZIP file that contains a folder named `chroma_db/` at the root. On first run the app downloads and unzips it next to the app.

If `chroma_db/` is missing and `CHROMA_DB_URL` is not set, the app will stop and ask you to provide it or build the DB using your pipeline.

## Deploy to Streamlit Cloud

- Push this repo to GitHub
- Create a new app, set the entrypoint to `streamlit_app_enhanced.py`
- Add the following Secrets:
  - `OPENAI_API_KEY`
  - `CHROMA_DB_URL` (link to a zipped `chroma_db/`)

## Notes

- `chroma_db/` is ignored via `.gitignore`. Build locally or supply a download URL via `CHROMA_DB_URL`.
