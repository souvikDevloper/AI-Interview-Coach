"""Professional Screen – Fireworks-only version
──────────────────────────────────────────────"""

# ── one-off process flags ──────────────────────────────────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  
os.environ["CT2_FORCE_CPU"]      = "1"   # CTranslate2  >= 4.3
os.environ["CUDA_VISIBLE_DEVICES"] = "" # OpenMP clash fix

# ── stdlib / 3-rd-party ───────────────────────────────────
import json
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
load_dotenv()                                           # pulls FIREWORKS_API_KEY

import streamlit as st
from streamlit_lottie import st_lottie
from audio_recorder_streamlit import audio_recorder

from langchain.memory import ConversationBufferWindowMemory
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import FAISS

import nltk
from prompts.prompts import templates
from speech_recognition.offline import save_wav_file, transcribe
from tts.edge_speak import speak

# ── constants ─────────────────────────────────────────────
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
EMBED_MODEL     = "nomic-embed-text"

# ── UI helpers ────────────────────────────────────────────
def load_lottiefile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

st_lottie(load_lottiefile("images/welcome.json"), loop=True, height=300)

with st.expander("Why did recording fail?"):
    st.write("Most likely the microphone didn’t record. "
             "Check mic connection and browser permission.")

# ── user inputs ───────────────────────────────────────────
jd        = st.text_area("Job description or keywords (e.g. *PostgreSQL*, *Python*):")
auto_play = st.checkbox("Let AI interviewer speak (Edge-TTS)")

@dataclass
class Message:
    origin : Literal["human", "ai"]
    message: str

# ── vector-store helpers ─────────────────────────────────
def build_retriever(text: str):
    nltk.download("punkt", quiet=True)
    chunks = NLTKTextSplitter().split_text(text)
    store  = FAISS.from_texts(
        chunks,
        FireworksEmbeddings(model_name=EMBED_MODEL)
    )
    return store.as_retriever(search_type="similarity")

# ── session-state bootstrap ──────────────────────────────
def init_state():
    if "retriever" not in st.session_state:
        st.session_state.retriever = build_retriever(jd)

    if "chain_kwargs" not in st.session_state:
        st.session_state.chain_kwargs = {
            "prompt": PromptTemplate(
                input_variables=["context", "question"],
                template=templates.jd_template)
        }

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
             k=6,
             human_prefix="Candidate",
             ai_prefix="Interviewer",
             return_messages=True       # required by ConversationChain ≥0.2.7
        )

    if "history" not in st.session_state:
        st.session_state.history = [
            Message("ai",
                    "Hello! I'm your professional interviewer. "
                    "Please introduce yourself (max 4 097 tokens per answer).")
        ]

    if "guideline" not in st.session_state:
        st.session_state.guideline = RetrievalQA.from_chain_type(
            llm=ChatFireworks(model_name=FIREWORKS_MODEL, temperature=0.8),
            chain_type="stuff",
            chain_type_kwargs=st.session_state.chain_kwargs,
            retriever=st.session_state.retriever,
            memory=st.session_state.memory
        ).invoke(
            {"query": "Create an interview guideline with ONE technical question per topic."}
        )["result"]

    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationChain(
            prompt=PromptTemplate(
                input_variables=["history", "input"],
                template=("Act as an interviewer following the guideline.\n"
                          "Ask **one** question, wait for answer, no explanations.\n\n"
                          "Conversation so far:\n{history}\n\nCandidate: {input}\nAI: ")
            ),
            llm=ChatFireworks(model_name=FIREWORKS_MODEL, temperature=0.8),
            memory=st.session_state.memory
        )

    if "feedback_chain" not in st.session_state:
        st.session_state.feedback_chain = ConversationChain(
            prompt=PromptTemplate(
                input_variables=["history", "input"],
                template=templates.feedback_template),
            llm=ChatFireworks(model_name=FIREWORKS_MODEL, temperature=0.5),
            memory=st.session_state.memory
        )

# ── answer handler ───────────────────────────────────────
def handle_answer(blob):
    # speech-to-text if voice mode
    if voice_mode:
        save_wav_file("temp/audio.wav", blob)
        try:
            user_text = transcribe("temp/audio.wav")
        except Exception:
            st.session_state.history.append(
                Message("ai", "Sorry, I couldn't understand that audio."))
            return None
    else:
        user_text = blob

    st.session_state.history.append(Message("human", user_text))
    reply = st.session_state.conversation.predict(input=user_text)
    st.session_state.history.append(Message("ai", reply))
    return speak(reply) if auto_play else None

# ── main layout ──────────────────────────────────────────
if jd:
    init_state()

    col1, col2 = st.columns(2)
    if col1.button("Show interview guideline"):
        st.write(st.session_state.guideline)
        st.stop()

    if col2.button("Get interview feedback"):
        fb = st.session_state.feedback_chain.predict(
            input="Provide an evaluation of the interview.")
        st.markdown(fb)
        st.download_button("Download feedback", fb,
                           file_name="interview_feedback.txt")
        st.stop()

    voice_mode = st.checkbox("Speak instead of typing")
    user_blob  = (audio_recorder(pause_threshold=2.5, sample_rate=44_100)
                  if voice_mode else st.chat_input("Your answer"))

    # wav_np is **always defined**, None if no answer yet
    wav_np = handle_answer(user_blob) if user_blob else None

    for m in st.session_state.history:
        role = "assistant" if m.origin == "ai" else "user"
        with st.chat_message(role):
            st.write(m.message)
            if role == "assistant" and auto_play and wav_np is not None:
                st.audio(wav_np, sample_rate=24_000)
else:
    st.info("Enter a job description to start the interview.")
