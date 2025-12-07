"""Professional Screen – Open LLM + HF embeddings"""

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

from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import FAISS

import nltk
from prompts.prompts import templates
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

# ── helpers ────────────────────────────────────────────────
def build_retriever(text: str):
    nltk.download("punkt", quiet=True)
    chunks = NLTKTextSplitter().split_text(text or "")
    store  = FAISS.from_texts(
        chunks,
        HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    )
    chunks = NLTKTextSplitter().split_text(text or "")
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
    if not qs:
        qs = fallback[:]
    return qs[:MAX_QUESTIONS]

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

# ── session bootstrap ──────────────────────────────────────
def init_state(jd: str):
    if "retriever" not in st.session_state:
        st.session_state.retriever = build_retriever(jd)

    if "chain_kwargs" not in st.session_state:
        st.session_state.chain_kwargs = {
            "prompt": PromptTemplate(
                input_variables=["context", "question"],
                template=templates.jd_template
            )
        }

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            k=4, human_prefix="Candidate", ai_prefix="Interviewer", return_messages=True
        )

    if "history" not in st.session_state:
        st.session_state.history = [
            Message("ai", "Welcome! Please introduce yourself briefly.")
        ]

    if "guideline" not in st.session_state:
        llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=800)
        st.session_state.guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
        llm = build_chat_model(temperature=0.3, context_window=800)
        st.session_state.guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            chain_type_kwargs=st.session_state.chain_kwargs,
            retriever=st.session_state.retriever,
            memory=st.session_state.memory
        ).invoke({"query": "Create a concise interview guideline with focused technical questions. Keep questions short."})["result"]

    if "questions" not in st.session_state:
        fallback = [
            "What recent project best showcases your {tech} skills?".format(tech="core tech"),
            "How do you debug performance bottlenecks in Python?",
            "Explain an SQL query you optimized—what changed?",
            "How do you design for reliability and observability?",
            "What trade-offs did you make in your last architecture?",
        ]
        st.session_state.questions = extract_questions(st.session_state.guideline, fallback)

    if "q_idx" not in st.session_state:
        st.session_state.q_idx = 0
    if "max_q" not in st.session_state:
        st.session_state.max_q = MAX_QUESTIONS
    if "finished" not in st.session_state:
        st.session_state.finished = False

    # feedback chain (kept for one-click report)
    if "feedback_llm" not in st.session_state:
        st.session_state.feedback_llm = ChatOllama(model=LLM_MODEL, temperature=0.2, num_predict=600)
    if "finished" not in st.session_state:
        st.session_state.finished = False

    # feedback chain (kept for one-click report)
    if "feedback_llm" not in st.session_state:
        st.session_state.feedback_llm = build_chat_model(temperature=0.2, context_window=600)

# ── turn handler ───────────────────────────────────────────
def handle_answer(blob, auto_play: bool):
    if st.session_state.finished:
        return None

    # STT if needed
    if st.session_state.voice_mode:
        save_wav_file("temp/audio.wav", blob)
        try:
            user_text = transcribe("temp/audio.wav")
        except Exception:
            st.session_state.history.append(Message("ai", "Sorry, I couldn't understand that audio."))
            return None
    else:
        user_text = blob

    # record candidate answer
    st.session_state.history.append(Message("human", user_text))

    # next question (deterministic; no LLM call ⇒ low latency)
    nxt = ask_next_question()
    if nxt is None:
        closing = "Thanks—that concludes the interview. You can click **Get feedback** to see your evaluation."
        st.session_state.history.append(Message("ai", closing))
        return speak(closing) if auto_play else None

    st.session_state.history.append(Message("ai", nxt))
    return speak(nxt) if auto_play else None

# ── UI ─────────────────────────────────────────────────────
st.title("Professional Interview")

jd = st.text_area("Job description / keywords (e.g., *PostgreSQL*, *Python*):")
auto_play = st.checkbox("Let interviewer speak (Edge-TTS)", value=False)

if jd:
    if not require_ollama():
        st.stop()

    init_state(jd)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Show guideline"):
            st.info(st.session_state.guideline)
    with c2:
        if st.button("Get feedback"):
            tmpl = PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template)
            prompt = tmpl.format(history=render_transcript(), input="Provide an evaluation of the interview.")
            fb = st.session_state.feedback_llm.invoke(prompt).content
            st.markdown(fb)
            st.download_button("Download feedback", fb, file_name="professional_feedback.txt")
    with c3:
        if st.button("Restart interview"):
            for k in ("history", "q_idx", "finished"):
                if k in st.session_state: del st.session_state[k]
            st.experimental_rerun()

    st.session_state.voice_mode = st.checkbox("Speak instead of typing", value=False)
    user_blob = audio_recorder(pause_threshold=2.5, sample_rate=44100) if st.session_state.voice_mode and not st.session_state.finished else (
                st.chat_input("Your answer") if not st.session_state.finished else None)

    wav_np = handle_answer(user_blob, auto_play) if user_blob else None

    # transcript
    for m in st.session_state.history:
        role = "assistant" if m.origin == "ai" else "user"
        with st.chat_message(role):
            st.write(m.message)
            if role == "assistant" and auto_play and wav_np is not None:
                st.audio(wav_np, sample_rate=24000)
else:
    st.info("Paste a job description to begin.")
