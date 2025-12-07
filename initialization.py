import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from prompts.prompts import templates
from langchain.prompts.prompt import PromptTemplate
from PyPDF2 import PdfReader
from prompts.prompt_selector import prompt_sector

""" 
This initializer now supports plug‑and‑play back‑ends:

• Speech‑to‑Text  — defaults to local faster‑whisper (edge of this repo)
• Speech synthesis — defaults to Edge‑TTS (no API key needed)
• LLM             — defaults to local Ollama (llama3:8b‑instruct‑q4_0)

Set these env vars (or Streamlit secrets) to override:
    VOICE_BACKEND = edge | local | azure | none
    LLM_BACKEND   = ollama | openai
    EDGE_VOICE    = a valid Edge voice name (e.g. en‑US‑AriaNeural)

If you keep the old cloud keys in place the original behaviour is unchanged.
"""

# ╭─ Back‑end selectors ─────────────────────────────────────────╮
VOICE_BACKEND = os.getenv("VOICE_BACKEND", "edge").lower()
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()

# ── Speech modules ──────────────────────────────────────────────
if VOICE_BACKEND == "edge":
    from speech_recognition.offline import transcribe_audio  # faster‑whisper wrapper
    from tts.edge_speak import speak                        # Edge‑TTS helper (online, free)
elif VOICE_BACKEND == "local":
    from speech_recognition.offline import transcribe_audio  # same STT
    from tts.piper import speak                             # Piper (fully offline)
else:
    from speech_recognition.azure import transcribe_audio    # fall back to original Azure path
    from tts.azure import speak

# ── LLM selector ───────────────────────────────────────────────
def _build_llm(temp: float):
    if LLM_BACKEND == "ollama":
        model = os.getenv("OLLAMA_MODEL", "llama3")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(base_url=base_url, model=model, temperature=temp)

    # lightweight local fallback using transformers via langchain_community
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_name = os.getenv("HF_MODEL", "sshleifer/tiny-gpt2")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    text_gen = pipeline("text-generation", model=mdl, tokenizer=tok, max_new_tokens=256, do_sample=True, temperature=temp)
    return HuggingFacePipeline(pipeline=text_gen)

# ╰───────────────────────────────────────────────────────────────╯


def embedding(text):
    """Split text and embed using open embeddings then return a FAISS store."""
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return FAISS.from_texts(texts, embeddings)


def resume_reader(resume):
    pdf_reader = PdfReader(resume)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def initialize_session_state(template=None, position=None):
    """Initialise all Streamlit session‑state objects the app relies on."""

    # ── Build vector store from JD or Resume ──
    if "jd" in st.session_state:
        st.session_state.docsearch = embedding(st.session_state.jd)
    else:
        st.session_state.docsearch = embedding(resume_reader(st.session_state.resume))

    st.session_state.retriever = st.session_state.docsearch.as_retriever(search_type="similarity")

    # ── Prompt(s) for QA chain ──
    if "jd" in st.session_state:
        interview_prompt = PromptTemplate(input_variables=["context", "question"], template=template)
        st.session_state.chain_type_kwargs = {"prompt": interview_prompt}
    else:
        st.session_state.chain_type_kwargs = prompt_sector(position, templates)

    # ── Memory & misc state vars ──
    st.session_state.memory = ConversationBufferMemory()
    st.session_state.history = []
    st.session_state.token_count = 0

    # ── Guideline chain (one‑shot) ──
    guideline_llm = _build_llm(temp=0.6)
    st.session_state.guideline = RetrievalQA.from_chain_type(
        llm=guideline_llm,
        chain_type_kwargs=st.session_state.chain_type_kwargs,
        chain_type="stuff",
        retriever=st.session_state.retriever,
        memory=st.session_state.memory,
    ).run(
        "Create an interview guideline and prepare only one question for each topic. Make sure the question tests the technical knowledge."
    )

    # ── Live interview chain ──
    screen_llm = _build_llm(temp=0.8)
    screen_prompt = PromptTemplate(
        input_variables=["history", "input"],
        template="""I want you to act as an interviewer strictly following the guideline in the current conversation.\n\nAsk me questions and wait for my answers like a real person. Do not write explanations. Ask one question at a time, do not repeat questions. Follow up if necessary. Your name is GPTInterviewer.\n\nCurrent Conversation:\n{history}\n\nCandidate: {input}\nAI: """,
    )
    st.session_state.screen = ConversationChain(prompt=screen_prompt, llm=screen_llm, memory=st.session_state.memory)

    # ── Feedback chain ──
    feedback_llm = _build_llm(temp=0.5)
    st.session_state.feedback = ConversationChain(
        prompt=PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template),
        llm=feedback_llm,
        memory=st.session_state.memory,
    )
