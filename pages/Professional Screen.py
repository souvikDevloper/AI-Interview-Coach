    

    # ── one-off flags ───────────────────────────────────────────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CT2_FORCE_CPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import hashlib
import tempfile
from dataclasses import dataclass
from typing import Literal, List

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from audio_recorder_streamlit import audio_recorder

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate



from prompts.prompts import templates
from speech_recognition.offline import transcribe
from tts.edge_speak import speak

GROQ_MODEL    = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_MODEL   = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_QUESTIONS = 12
RETRIEVE_K    = int(os.getenv("RETRIEVE_K", "4"))

@dataclass
class Message:
    origin : Literal["human", "ai"]
    message: str

def _mk_llm(temperature: float, max_tokens: int):
    try:
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, max_tokens=max_tokens)
    except TypeError:
        return ChatGroq(model_name=GROQ_MODEL, temperature=temperature, max_tokens=max_tokens)

def _invoke_chat(llm, prompt: str) -> str:
    r = llm.invoke(prompt)
    return getattr(r, "content", str(r))

def _bytes_to_temp_wav(raw_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(raw_bytes)
        f.flush()
        return f.name

def _render_transcript(hist: List[Message]) -> str:
    lines = []
    for m in hist:
        role = "Interviewer" if m.origin == "ai" else "Candidate"
        lines.append(role + ": " + m.message)
    return "\n".join(lines)

_q_line = re.compile(r"^\s*(?:[-*•]|\d+[\).:])?\s*(.+?)\s*\??\s*$")

def _extract_questions(guideline: str, fallback: List[str]) -> List[str]:
    qs: List[str] = []
    for raw in (guideline or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if "?" in line or line.lower().startswith(("q:", "question", "ask", "tell me", "describe")):
            m = _q_line.match(line)
            if not m:
                continue
            q = m.group(1).strip()
            if len(q) < 8:
                continue
            if not q.endswith("?"):
                q += "?"
            qs.append(q)
    return qs or fallback

def _split_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if chunk_size <= overlap:
        overlap = 0
    out = []
    i = 0
    n = len(t)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(t[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out

def _build_retriever(text: str):
    chunks = _split_text(text)
    if not chunks:
        chunks = ["(empty)"]
    vs = FAISS.from_texts(chunks, HuggingFaceEmbeddings(model_name=EMBED_MODEL))
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVE_K})

def _retrieve_context(retriever, query: str) -> str:
    # works across retriever API versions
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)
    parts = []
    for d in docs or []:
        pc = getattr(d, "page_content", None)
        if pc:
            parts.append(pc)
    return "\n\n".join(parts)[:12000]  # cap

def _make_feedback_report(llm, hist: List[Message]) -> str:
    tmpl = PromptTemplate(input_variables=["history", "input"], template=templates.feedback_template)
    prompt = tmpl.format(
        history=_render_transcript(hist),
        input="Provide an evaluation of the interview. Include score, strengths, gaps, improved answers, and a short 7-day plan."
    )
    return _invoke_chat(llm, prompt)


def init_state(jd: str):
        st.session_state.source_text = jd
        st.session_state.setdefault("history", [Message("ai", "Welcome! Please introduce yourself briefly.")])
        st.session_state.setdefault("q_idx", 0)
        st.session_state.setdefault("finished", False)
        st.session_state.setdefault("report_text", "")
        st.session_state.setdefault("last_audio_hash", "")

        if "retriever" not in st.session_state:
            st.session_state.retriever = _build_retriever(jd)

        if "llm" not in st.session_state:
            st.session_state.llm = _mk_llm(temperature=0.3, max_tokens=900)

        if "feedback_llm" not in st.session_state:
            st.session_state.feedback_llm = _mk_llm(temperature=0.2, max_tokens=900)

        if "guideline" not in st.session_state:
            query = "Create a concise interview guideline with focused technical questions. Keep questions short."
            ctx = _retrieve_context(st.session_state.retriever, query)
            tmpl = PromptTemplate(input_variables=["context", "question"], template=templates.jd_template)
            prompt = tmpl.format(context=ctx, question=query)
            try:
                st.session_state.guideline = _invoke_chat(st.session_state.llm, prompt)
            except Exception:
                st.session_state.guideline = ""

        if "questions" not in st.session_state:
            fallback = [
                "Walk me through your most impactful recent project.",
                "How do you debug a production issue end-to-end?",
                "Explain a system design trade-off you made recently.",
                "How do you optimize a slow SQL query?",
                "How do you handle retries and idempotency in distributed systems?",
            ]
            st.session_state.questions = _extract_questions(st.session_state.guideline, fallback)

def _handle_input(blob, auto_play: bool):
        if st.session_state.finished or blob is None:
            return None

        if isinstance(blob, bytes):
            h = hashlib.sha1(blob).hexdigest()
            if st.session_state.get("last_audio_hash") == h:
                return None
            st.session_state["last_audio_hash"] = h
            wav_path = _bytes_to_temp_wav(blob)
            user_text = transcribe(wav_path)
        else:
            user_text = blob

        user_text = (user_text or "").strip()
        if not user_text:
            return None

        st.session_state.history.append(Message("human", user_text))

        q_idx = st.session_state.q_idx
        if q_idx < len(st.session_state.questions) and q_idx < MAX_QUESTIONS:
            next_q = st.session_state.questions[q_idx]
            st.session_state.q_idx += 1
        else:
            st.session_state.finished = True
            next_q = "Thanks — that concludes the technical interview."

        st.session_state.history.append(Message("ai", next_q))

        if auto_play:
            return speak(next_q)
        return None

    # ── UI ─────────────────────────────────────────────────────
st.set_page_config(page_title="Professional Screen", page_icon="💻")
st.title("💻 Professional / Technical Interview")

jd = st.text_area("Paste the job description here (used to tailor questions)", height=220, placeholder="Paste the JD…")

if jd.strip():
        init_state(jd)

        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            auto_play = st.checkbox("Auto-play questions (TTS)", value=False)
            with st.expander("Show generated guideline"):
                st.write(st.session_state.get("guideline", ""))
        with c2:
            if st.button("Get feedback"):
                st.session_state.report_text = _make_feedback_report(st.session_state.feedback_llm, st.session_state.history)
        with c3:
            if st.button("Restart interview"):
                for k in ("history","q_idx","finished","report_text","last_audio_hash","retriever","guideline","questions","llm","feedback_llm","source_text"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        if st.session_state.get("report_text"):
            st.download_button("Download report.txt", st.session_state.report_text, file_name="professional_report.txt", mime="text/plain")
            with st.expander("View report"):
                st.text_area("Feedback", st.session_state.report_text, height=350)

        voice_mode = st.checkbox("Speak instead of typing", value=False)

        user_blob = (
            audio_recorder(pause_threshold=2.5, sample_rate=44100)
            if voice_mode and not st.session_state.finished
            else (st.chat_input("Your answer") if not st.session_state.finished else None)
        )

        wav_np = _handle_input(user_blob, auto_play) if user_blob else None

        for m in st.session_state.history:
            with st.chat_message("assistant" if m.origin == "ai" else "user"):
                st.write(m.message)

        if auto_play and wav_np is not None:
            st.audio(wav_np, sample_rate=24000)
else:
        st.info("Paste a job description to begin.")
