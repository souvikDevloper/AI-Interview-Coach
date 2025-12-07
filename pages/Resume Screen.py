"""Resume Screen – local, open-source models via Ollama"""
"""Resume Screen – Open LLM + HF embeddings"""

# ── one-off flags ───────────────────────────────────────────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CT2_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ── stdlib / 3rd-party ─────────────────────────────────────
import re
from dataclasses import dataclass
from typing import Literal, List

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from PyPDF2 import PdfReader

from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import FAISS

import nltk
from prompts.prompts import templates
from prompts.prompt_selector import prompt_sector
from speech_recognition.offline import save_wav_file, transcribe
from tts.edge_speak import speak
from app_utils import require_ollama

# ── constants ──────────────────────────────────────────────
LLM_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
from app_utils import build_chat_model, build_embeddings

# ── constants ──────────────────────────────────────────────
MAX_QUESTIONS   = 12

@dataclass
class Message:
    origin : Literal["human", "ai"]
    message: str

def build_retriever(pdf_file):
    nltk.download("punkt", quiet=True)
    text = "".join(p.extract_text() or "" for p in PdfReader(pdf_file).pages)
    chunks = NLTKTextSplitter().split_text(text)
    store  = FAISS.from_texts(
        chunks,
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    chunks = NLTKTextSplitter().split_text(text)
    store  = FAISS.from_texts(chunks, build_embeddings())
    return store.as_retriever(search_type="similarity")

_q_line = re.compile(r"""^\s*(?:[-*•]|\d+[\).:])?\s*(.+?)\s*\??\s*$""")
def extract_questions(guideline: str, fallback: List[str]) -> List[str]:
    qs: List[str] = []
    for raw in (guideline or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "?" in line or line.lower().startswith(("q:", "question", "ask")):
            m = _q_line.match(line)
            if m:
                q = m.group(1)
                if len(q) >= 8:
                    if not q.endswith("?"):
                        q += "?"
                    qs.append(q)
    return (qs or fallback)[:MAX_QUESTIONS]

def ask_next_question() -> str | None:
    i   = st.session_state.q_idx
    qs  = st.session_state.questions
    cap = st.session_state.max_q
    if i >= len(qs) or i >= cap:
        st.session_state.finished = True
        return None
    q = qs[i]
    st.session_state.q_idx += 1
    return q

def render_transcript() -> str:
    lines = []
    for m in st.session_state.history:
        who = "Interviewer" if m.origin == "ai" else "Candidate"
        lines.append(f"{who}: {m.message}")
    return "\n".join(lines)

def init_state(position: str, pdf):
    if "retriever" not in st.session_state:
        st.session_state.retriever = build_retriever(pdf)

    if "chain_kwargs" not in st.session_state:
        st.session_state.chain_kwargs = prompt_sector(position, templates)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=4, human_prefix="Candidate", ai_prefix="Interviewer", return_messages=True
        )

    if "history" not in st.session_state:
        st.session_state.history = [Message("ai", "Hello! Let’s discuss your resume. Give a brief intro.")]

    if "guideline" not in st.session_state:
        llm = build_chat_model(temperature=0.3, context_window=800)
        st.session_state.guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs=st.session_state.chain_kwargs,
            retriever=st.session_state.retriever,
            memory=st.session_state.memory
        ).invoke({"query": "Create an interview guideline focused on the uploaded resume. Keep questions short."})["result"]

    if "questions" not in st.session_state:
        fallback = [
            "Walk me through the most impactful project on your resume.",
            "Which achievement are you proudest of and why?",
            "Tell me about a tough bug you fixed—root cause and fix?",
            "What’s a metric you improved and how?",
            "Which part of your experience best fits this role?",
        ]
        st.session_state.questions = extract_questions(st.session_state.guideline, fallback)

    if "q_idx" not in st.session_state:   st.session_state.q_idx = 0
    if "max_q" not in st.session_state:   st.session_state.max_q = MAX_QUESTIONS
    if "finished" not in st.session_state:st.session_state.finished = False

    if "feedback_llm" not in st.session_state:
        st.session_state.feedback_llm = build_chat_model(temperature=0.2, context_window=600)

def handle_answer(blob, auto_play: bool):
    if st.session_state.finished:
        return None

    if st.session_state.voice_mode:
        save_wav_file("temp.wav", blob)
        try:
            user_text = transcribe("temp.wav")
        except Exception:
            st.session_state.history.append(Message("ai", "Sorry, I couldn't understand that audio."))
            return None
    else:
        user_text = blob

    st.session_state.history.append(Message("human", user_text))

    nxt = ask_next_question()
    if nxt is None:
        closing = "Thanks—that concludes the resume round. Click **Get feedback** for your evaluation."
        st.session_state.history.append(Message("ai", closing))
        return speak(closing) if auto_play else None

    st.session_state.history.append(Message("ai", nxt))
    return speak(nxt) if auto_play else None

# ── UI ─────────────────────────────────────────────────────
st.title("Resume Interview")

position  = st.selectbox("Position", ["Data Analyst","Software Engineer","Marketing"])
resume_pdf= st.file_uploader("Upload your resume (PDF)", type=["pdf"])
auto_play = st.checkbox("Let interviewer speak (Edge-TTS)", value=False)

if position and resume_pdf:
    init_state(position, resume_pdf)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Show guideline"):
            st.info(st.session_state.guideline)
    with c2:
        if st.button("Get feedback"):
            tmpl = PromptTemplate(input_variables=["history","input"], template=templates.feedback_template)
            fb   = st.session_state.feedback_llm.invoke(tmpl.format(history=render_transcript(), input="Evaluate the interview."))
            st.markdown(fb)
            st.download_button("Download feedback", fb, file_name="resume_feedback.txt")
    with c3:
        if st.button("Restart interview"):
            for k in ("history","q_idx","finished"):
                if k in st.session_state: del st.session_state[k]
            st.experimental_rerun()

    st.session_state.voice_mode = st.checkbox("Speak instead of typing", value=False)
    inp = audio_recorder(pause_threshold=2.0, sample_rate=44100) if st.session_state.voice_mode and not st.session_state.finished else (
          st.chat_input("Your answer") if not st.session_state.finished else None)

    wav_np = handle_answer(inp, auto_play) if inp else None

    for m in st.session_state.history:
        role = "assistant" if m.origin == "ai" else "user"
        with st.chat_message(role):
            st.write(m.message)
            if role == "assistant" and auto_play and wav_np is not None:
                st.audio(wav_np, sample_rate=24000)
else:
    st.info("Select a position and upload a PDF to begin.")
