import os
import json
from typing import List, Dict, Any
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# ----- Environment & Constants -----
load_dotenv()
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")

APP_DIR = os.path.dirname(__file__)
PROJECT_ROOT = APP_DIR  # Since we're already in the project root
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
DEFAULT_COLLECTION = "multimodal_docs"

# ----- Enhanced Prompt -----
RAG_PROMPT = """
You are a medical assistant specialized in endocrinology, particularly in Hashimoto's disease.
Your task is to answer the user's medical question using only the retrieved context provided.

Guidelines:
- Base your answer strictly on the retrieved context.
- Do not add new examples, lists, or numerical values unless they appear in the retrieved context.
- If the retrieved context is insufficient, clearly state this and provide only general, cautious background without making up specifics.
- Avoid giving specific medical or dietary recommendations unless they are directly stated in the retrieved context.
- Keep responses concise, medically accurate, and easy to understand.
- Structure your answer with headings and bullet points when appropriate.

Sources section rules:
- For normal text/PDF files → include only the document filename(s) and page number(s) that were primarily used to create your answer.
- For WebVTT transcript files (filenames ending with `_vtt.txt`) →
  - Extract all timestamp ranges used.
  - Determine the earliest start timestamp and the latest end timestamp.
  - Display the source as:
      filename_vtt.txt | earliest_start_timestamp | latest_end_timestamp
  - Example:
      40-sleep-apnea-and-hashimotos-the-surprising-link-and-root-cause-solutions-beyond-cpap-dr-dylan-petkus_vtt.txt | 00:00:00.000 | 00:08:54.280

- If multiple documents were retrieved, choose only the one(s) with the most relevant information, not all.
- If no relevant information was found, write "No relevant sources found."

User Question:
{question}

Retrieved Medical Context:
{context}

Answer:

Sources:
"""





def get_api_key() -> str:
    """Resolve OpenAI API key from session, secrets, or environment."""
    key_from_session = st.session_state.get("openai_api_key", "")
    if key_from_session:
        return key_from_session
    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key_from_secrets = ""
    if key_from_secrets:
        return key_from_secrets
    return os.getenv("OPENAI_API_KEY", "")

# ----- Cached resources -----
@st.cache_resource(show_spinner=False)
def get_vectorstore(
    persist_directory: str,
    collection_name: str,
    embedding_model_name: str = "text-embedding-3-small",
    api_key: str = "",
) -> Chroma:
    """Get cached vector store instance bound to the provided API key."""
    embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=api_key or None)
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

def format_docs(docs) -> str:
    """Format documents with source information."""
    formatted_texts: List[str] = []
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown file")
        page_num = doc.metadata.get("page_number", "Unknown page")
        category = doc.metadata.get("category", "")
        source_info = f"\n\nSource: {filename} (Page {page_num})"
        if category:
            source_info += f" [{category}]"
        formatted_texts.append((doc.page_content or "") + source_info)
    return "\n\n".join(formatted_texts)

def build_chain(retriever, model_name: str, temperature: float, api_key: str = ""):
    """Build the RAG chain."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key or None)
    chain = (
        {
            "context": (retriever | format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )
    return chain

def get_collection_stats(vectorstore: Chroma) -> Dict[str, Any]:
    """Get collection statistics."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        return {
            "total_documents": count,
            "collection_name": collection.name,
            "embedding_function": "text-embedding-3-small"
        }
    except Exception as e:
        return {"error": str(e)}

def load_sample_questions() -> List[str]:
    """Load sample questions for quick testing."""
    return [
        "What are the common symptoms of Hashimoto's disease?",
        "What foods should be avoided in a Hashimoto's diet?",
        "How does Hashimoto's affect thyroid function?",
        "What are the recommended supplements for Hashimoto's?",
        "What lifestyle changes can help manage Hashimoto's symptoms?"
    ]

def initialize_session_state():
    """Initialize session state for conversation history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

def main():
    st.set_page_config(
        page_title="MediChat - Hashimoto's AI", 
        page_icon="🏥", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS for ChatGPT-like styling
    st.markdown("""
    <style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container - compact but with sidebar space */
    .main {
        padding: 0 !important;
        margin-left: 0 !important;
        max-width: 1000px !important;
    }
    
    /* ChatGPT-like header */
    .chat-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        text-align: center;
        border-bottom: 1px solid #e5e5e5;
        margin-bottom: 0;
        border-radius: 0 0 10px 10px;
    }
    
    /* Message containers - more compact */
    .message-container {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #f0f0f0;
        max-width: 100%;
    }
    
    .user-message {
        background: #f7f7f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        max-width: 100%;
        word-wrap: break-word;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 2px solid #e8f2ff;
        max-width: 100%;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
        position: relative;
    }
    
    .assistant-message::before {
        content: "🏥";
        position: absolute;
        top: -10px;
        left: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message h1, .assistant-message h2, .assistant-message h3 {
        color: #2c3e50;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .assistant-message ul, .assistant-message ol {
        background: rgba(102, 126, 234, 0.05);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .assistant-message li {
        margin: 0.5rem 0;
        color: #34495e;
    }
    
    .assistant-message strong {
        color: #667eea;
        font-weight: 600;
    }
    
    .assistant-message .sources-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        border: none;
    }
    
    /* Input area - adjusted for sidebar */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #e5e5e5;
        padding: 1rem 1.5rem;
        z-index: 1000;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sample questions - more compact */
    .sample-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
        justify-content: center;
    }
    
    .sample-btn {
        background: #f0f0f0;
        border: 1px solid #d0d0d0;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: all 0.2s;
        max-width: 300px;
    }
    
    .sample-btn:hover {
        background: #e0e0e0;
    }
    
    /* Sources section */
    .sources-section {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #667eea;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .message-container {
            padding: 1rem;
        }
        .input-container {
            padding: 1rem;
        }
        .main {
            max-width: 100% !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="chat-header"><h1>🏥 MediChat - Hashimoto\'s AI</h1><p>Your intelligent medical assistant for Hashimoto\'s disease</p></div>', unsafe_allow_html=True)

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API key management (UI + secrets/env)
        st.subheader("🔑 OpenAI API Key")
        current_key_present = bool(get_api_key())
        if current_key_present:
            st.success("API key is set (session/secrets/env)")
        with st.expander("Set/Update key (stored only for this session)", expanded=not current_key_present):
            input_key = st.text_input("Enter OpenAI API key", type="password", placeholder="sk-...", help="Stored only in session state, not persisted.")
            colk1, colk2 = st.columns([1,1])
            with colk1:
                if st.button("Save key", use_container_width=True):
                    if input_key.strip():
                        st.session_state["openai_api_key"] = input_key.strip()
                        os.environ["OPENAI_API_KEY"] = input_key.strip()
                        st.rerun()
            with colk2:
                if st.button("Clear session key", use_container_width=True):
                    if "openai_api_key" in st.session_state:
                        del st.session_state["openai_api_key"]
                    # Do not unset env if it came from secrets; only unset if we set it
                    # Best-effort: if env matches session-cleared value, remove it
                    # (no-op if different source)
                    # Note: Avoid logging key values
                    st.rerun()

        st.divider()
        
        # Model settings (simplified)
        st.subheader("🤖 Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1, help="Higher = more creative, Lower = more focused")
        top_k = st.slider("Top-K Documents", 1, 15, st.session_state.top_k, help="Number of documents to retrieve")
        stream_resp = st.checkbox("Stream response", value=True, help="Show response as it's being generated")
        
        # Update session state
        st.session_state.temperature = temperature
        st.session_state.top_k = top_k
        
        # Quick settings
        st.divider()
        st.subheader("⚡ Quick Settings")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎯 Precise", help="Low temperature, focused responses"):
                st.session_state.temperature = 0.0
                st.session_state.top_k = 3
                st.rerun()
        with col2:
            if st.button("🧠 Creative", help="Higher temperature, more creative responses"):
                st.session_state.temperature = 0.7
                st.session_state.top_k = 8
                st.rerun()
        
        st.divider()
        
        # Database info (simplified)
        st.subheader("🗄️ Database")
        if os.path.isdir(DB_DIR):
            try:
                vectorstore = get_vectorstore(DB_DIR, DEFAULT_COLLECTION, api_key=get_api_key())
                stats = get_collection_stats(vectorstore)
                if "error" not in stats:
                    st.metric("Total Documents", stats["total_documents"])
                else:
                    st.error(f"Error: {stats['error']}")
            except Exception as e:
                st.error(f"DB Error: {e}")
        else:
            st.error("❌ Vector DB not found")

    # Main chat area
    main_container = st.container()
    
    with main_container:
        # Display conversation history
        if st.session_state.messages:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="message-container"><div class="user-message"><strong>You:</strong> {message["content"]}</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="message-container"><div class="assistant-message"><strong>MediChat:</strong> {message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            # Welcome message and sample questions
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #666;">
                <h2>👋 Welcome to MediChat!</h2>
                <p>Ask me anything about Hashimoto's disease, symptoms, treatments, or diet recommendations.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Sample questions
            st.subheader("💡 Try asking:")
            sample_questions = load_sample_questions()
            cols = st.columns(2)
            for i, question in enumerate(sample_questions):
                col_idx = i % 2
                if cols[col_idx].button(f"{question}", key=f"sample_{i}", use_container_width=True):
                    st.session_state["sample_question"] = question
                    st.rerun()

    # Input area (fixed at bottom)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
        # Question input with form
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Your question", 
            value=st.session_state.get("sample_question", ""),
            placeholder="Ask me anything about Hashimoto's disease...",
            height=80,
            help="Ask any question about Hashimoto's disease, diet, symptoms, or treatments"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            ask = st.form_submit_button("🩺 Ask", type="primary", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🗑️ Clear Chat", use_container_width=True)
            if clear:
                st.session_state.messages = []
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Process question
    if ask and question.strip():
        # Add user question to conversation history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Clear the sample question after using it
        if "sample_question" in st.session_state:
            del st.session_state["sample_question"]
        

        
        # Display user question immediately
        st.markdown(f'<div class="message-container"><div class="user-message"><strong>You:</strong> {question}</div></div>', unsafe_allow_html=True)
        
        # Check prerequisites
        if not os.path.isdir(DB_DIR):
            st.error("❌ Vector DB not found. Please run the pipeline to build and persist the Chroma DB first.")
            st.info("💡 Run `python main.py` to process your PDFs and create the vector database.")
            return

        resolved_api_key = get_api_key()
        if not resolved_api_key:
            st.warning("⚠️ Add your OpenAI API key in the sidebar to proceed.")
            return

        # Initialize vector store and chain
        try:
            vectorstore = get_vectorstore(DB_DIR, DEFAULT_COLLECTION, api_key=resolved_api_key)
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
            chain = build_chain(retriever, model_name="gpt-4o-mini", temperature=temperature, api_key=resolved_api_key)
            
            # Generate response
            if stream_resp:
                placeholder = st.empty()
                parts: List[str] = []
                with st.spinner("🤔 Thinking..."):
                    for chunk in chain.stream(question):
                        text_piece = ""
                        if isinstance(chunk, str):
                            text_piece = chunk
                        else:
                            content_attr = getattr(chunk, "content", None)
                            if isinstance(content_attr, str):
                                text_piece = content_attr
                            else:
                                text_attr = getattr(chunk, "text", None)
                                if isinstance(text_attr, str):
                                    text_piece = text_attr
                        if text_piece:
                            parts.append(text_piece)
                            placeholder.markdown(f'<div class="message-container"><div class="assistant-message">{"".join(parts)}</div></div>')
                final_text = "".join(parts)
                # print('='*50)
                # print(final_text)
                # print('='*50)
            else:
                with st.spinner("🤔 Thinking..."):
                    result = chain.invoke(question)
                    final_text = result.content
                    st.markdown(f'<div class="message-container"><div class="assistant-message">{final_text}</div></div>')
            
            # Add response to conversation history
            st.session_state.messages.append({"role": "assistant", "content": final_text})
            

                    
        except Exception as exc:
            error_msg = f"❌ Error while generating answer: {exc}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

if __name__ == "__main__":
    main()