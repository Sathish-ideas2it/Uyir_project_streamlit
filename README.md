# Uyir_project_streamlit

Streamlit app for a Hashimoto's RAG assistant using ChromaDB and OpenAI via LangChain.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app_enhanced.py
```

Set `OPENAI_API_KEY` in your environment or via Streamlit secrets.

## Deploy to Streamlit Cloud

- Push this repo to GitHub
- In Streamlit Cloud, set the app entrypoint to `streamlit_app_enhanced.py`
- Add a secret `OPENAI_API_KEY`

## Notes

- The `chroma_db/` directory is ignored to keep the repo small. Build it in your environment before running.
