"""
capstone_streamlit.py — Streamlit UI for PhysicsBuddy Agent
Fixed version with step-by-step loading and proper error handling.

Run with: streamlit run capstone_streamlit.py
"""

import streamlit as st
import uuid
import os

st.set_page_config(
    page_title="PhysicsBuddy — B.Tech Study Assistant",
    page_icon="⚛️",
    layout="wide"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚛️ PhysicsBuddy")
    st.markdown("**AI Study Assistant for B.Tech Physics**")
    st.markdown("---")
    st.markdown("### 📚 Topics Covered")
    for t in [
        "Newton's Laws of Motion", "Kinematics & Equations of Motion",
        "Work, Energy, and Power", "Laws of Thermodynamics",
        "Electrostatics", "Current Electricity & Circuits",
        "Waves and Sound", "Optics (Reflection & Refraction)",
        "Gravitation & Orbital Motion", "Modern Physics & Quantum Basics",
        "Rotational Motion", "Magnetism & EMI",
    ]:
        st.markdown(f"• {t}")
    st.markdown("---")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    if "thread_id" in st.session_state:
        st.caption(f"Session: {st.session_state.thread_id[:8]}...")
    st.markdown("---")
    st.caption("PhysicsBuddy v1.0 | Agentic AI Course 2026")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("⚛️ PhysicsBuddy — B.Tech Physics Study Assistant")
st.markdown(
    "Ask me anything about your B.Tech Physics syllabus — concepts, formulas, "
    "numericals, or exam tips. I remember our conversation throughout this session!"
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "thread_id"   not in st.session_state: st.session_state.thread_id   = str(uuid.uuid4())
if "app"         not in st.session_state: st.session_state.app         = None
if "load_error"  not in st.session_state: st.session_state.load_error  = None
if "loading"     not in st.session_state: st.session_state.loading     = False


# ── Cached resource loaders (each cached independently) ───────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY", "")
    )

@st.cache_resource(show_spinner=False)
def get_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_collection(_embedder):
    import chromadb
    from agent.knowledge_base import PHYSICS_DOCUMENTS
    client = chromadb.Client()
    try:
        client.delete_collection("physics_kb")
    except Exception:
        pass
    col = client.create_collection("physics_kb", metadata={"hnsw:space": "cosine"})
    texts     = [d["text"]          for d in PHYSICS_DOCUMENTS]
    ids       = [d["id"]            for d in PHYSICS_DOCUMENTS]
    metadatas = [{"topic": d["topic"]} for d in PHYSICS_DOCUMENTS]
    col.add(
        documents=texts,
        embeddings=_embedder.encode(texts).tolist(),
        ids=ids,
        metadatas=metadatas
    )
    return col

@st.cache_resource(show_spinner=False)
def get_app(_llm, _embedder, _col):
    from agent.graph import build_graph
    app, _, _, _ = build_graph(llm=_llm, embedder=_embedder, collection=_col)
    return app


# ── Load agent if not yet loaded ──────────────────────────────────────────────
if st.session_state.app is None and st.session_state.load_error is None:

    st.info("⏳ **First-time setup** — loading the AI agent. This takes 30–90 seconds "
            "because it downloads a 90MB language model. Subsequent runs are instant.")

    try:
        with st.spinner("1/4 Connecting to Groq..."):
            llm = get_llm()
        st.success("✅ Groq LLM connected")

        with st.spinner("2/4 Loading sentence embedder (downloading ~90MB on first run)..."):
            embedder = get_embedder()
        st.success("✅ Sentence embedder loaded")

        with st.spinner("3/4 Building physics knowledge base in ChromaDB..."):
            col = get_collection(embedder)
        st.success(f"✅ Knowledge base ready — {col.count()} documents indexed")

        with st.spinner("4/4 Compiling LangGraph agent..."):
            app = get_app(llm, embedder, col)
        st.success("✅ Agent compiled and ready!")

        st.session_state.app = app
        st.rerun()

    except Exception as e:
        st.session_state.load_error = str(e)
        st.rerun()


# ── Error state ───────────────────────────────────────────────────────────────
if st.session_state.load_error:
    st.error(f"**Loading failed:** {st.session_state.load_error}")
    st.markdown("""
**Fix checklist:**
1. Make sure API key is set — in PowerShell: `$env:GROQ_API_KEY = "your_key_here"`
2. Restart the terminal after setting the key, then run streamlit again
3. Make sure internet is connected (needed to download the model on first run)
4. Run `pip install sentence-transformers chromadb langgraph langchain-groq` if any package is missing
""")
    if st.button("🔄 Clear cache and retry"):
        st.cache_resource.clear()
        st.session_state.load_error = None
        st.rerun()
    st.stop()


# ── Wait for load ─────────────────────────────────────────────────────────────
if st.session_state.app is None:
    st.stop()


# ── App is ready — show chat ──────────────────────────────────────────────────
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "⚛️"):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a physics question..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚛️"):
        with st.spinner("Thinking..."):
            try:
                from agent.state import CapstoneState
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                init = CapstoneState(
                    question=prompt, messages=[], route="", retrieved="",
                    sources=[], tool_result="", answer="",
                    faithfulness=0.0, eval_retries=0, user_name="", topic_asked=""
                )
                result = st.session_state.app.invoke(init, config=config)

                answer  = result.get("answer", "Sorry, I could not generate a response.")
                sources = result.get("sources", [])
                route   = result.get("route", "")
                faith   = result.get("faithfulness", 0.0)

                reply = answer
                if sources:
                    reply += f"\n\n---\n📚 **Sources:** {', '.join(sources)}"

                st.markdown(reply)

                with st.expander("🔍 Agent trace", expanded=False):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Route",       route)
                    c2.metric("Faithfulness", f"{faith:.2f}")
                    c3.metric("Retries",      result.get("eval_retries", 0))

                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                err = f"⚠️ Error generating response: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})