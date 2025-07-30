"""Resume Screen – Fireworks-only stack
──────────────────────────────────────"""

# ── one-off flag ──────────────────────────────────────────────
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE") 
os.environ["CT2_FORCE_CPU"]      = "1"   # CTranslate2  >= 4.3
os.environ["CUDA_VISIBLE_DEVICES"] = "" # OpenMP clash fix

# ── stdlib / 3-rd-party ──────────────────────────────────────
import json
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
load_dotenv()                                           # FIREWORKS_API_KEY

import streamlit as st
from streamlit_lottie import st_lottie
from audio_recorder_streamlit import audio_recorder
from PyPDF2 import PdfReader

from langchain.memory import ConversationBufferWindowMemory
from langchain_fireworks import ChatFireworks, FireworksEmbeddings
from langchain.chains import ConversationChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import FAISS

import nltk
from prompts.prompts import templates
from prompts.prompt_selector import prompt_sector
from speech_recognition.offline import save_wav_file, transcribe
from tts.edge_speak import speak

# ── constants ────────────────────────────────────────────────
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
EMBED_MODEL     = "nomic-embed-text"

# ── UI helpers ───────────────────────────────────────────────
def load_lottiefile(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

st_lottie(load_lottiefile("images/welcome.json"), loop=True, height=300)

st.expander("Why did recording fail?").write("Check microphone connection and browser permissions.")
st.expander("Why did my resume upload fail?").write("Currently only **PDF** resumes are supported.")

# ── user inputs ──────────────────────────────────────────────
position   = st.selectbox("Position you are applying for",
                          ["Data Analyst", "Software Engineer", "Marketing"])
resume_pdf = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
auto_play  = st.checkbox("Let AI interviewer speak (Edge-TTS)")

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# ── PDF → retriever ──────────────────────────────────────────
def build_retriever(pdf_file) -> FAISS:
    nltk.download("punkt", quiet=True)
    text   = "".join(p.extract_text() for p in PdfReader(pdf_file).pages)
    chunks = NLTKTextSplitter().split_text(text)
    store  = FAISS.from_texts(chunks,
                              FireworksEmbeddings(model_name=EMBED_MODEL))
    return store.as_retriever(search_type="similarity")

# ── session-state bootstrap ─────────────────────────────────
def init_state():
    if "retriever" not in st.session_state:
        st.session_state.retriever = build_retriever(resume_pdf)

    if "chain_kwargs" not in st.session_state:
        st.session_state.chain_kwargs = prompt_sector(position, templates)

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
             k=6,
             human_prefix="Candidate",
             ai_prefix="Interviewer",
             return_messages=True       # required by ConversationChain ≥0.2.7
        )
    if "history" not in st.session_state:
        st.session_state.history = [
            Message("ai", "Hello! I’ll ask questions based on your resume. Please introduce yourself.")
        ]

    if "guideline" not in st.session_state:
        st.session_state.guideline = RetrievalQA.from_chain_type(
            llm=ChatFireworks(model_name=FIREWORKS_MODEL, temperature=0.8),
            chain_type="stuff",
            chain_type_kwargs=st.session_state.chain_kwargs,
            retriever=st.session_state.retriever,
            memory=st.session_state.memory
        ).invoke({"query": "Create an interview guideline; provide two questions per topic."})["result"]

    if "conversation" not in st.session_state:
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=("You are GPTInterviewer. Ask **one** question at a time based on the guideline.\n\n"
                      "{history}\n\nCandidate: {input}\nAI: "))
        st.session_state.conversation = ConversationChain(
            prompt=prompt,
            llm=ChatFireworks(model_name=FIREWORKS_MODEL, temperature=0.7),
            memory=st.session_state.memory)

    if "feedback_chain" not in st.session_state:
        st.session_state.feedback_chain = ConversationChain(
            prompt=PromptTemplate(
                input_variables=["history", "input"],
                template=templates.feedback_template),
            llm=ChatFireworks(model_name=FIREWORKS_MODEL, temperature=0.5),
            memory=st.session_state.memory)

# ── answer handler ───────────────────────────────────────────
def handle_answer(blob):
    user_text = (transcribe("temp.wav")
                 if voice_mode and save_wav_file("temp.wav", blob) else blob)
    st.session_state.history.append(Message("human", user_text))
    reply = st.session_state.conversation.predict(input=user_text)
    st.session_state.history.append(Message("ai", reply))
    return speak(reply) if auto_play else None

# ── main layout ──────────────────────────────────────────────
if position and resume_pdf:
    init_state()

    col1, col2 = st.columns(2)
    if col2.button("Show interview guideline"):
        st.markdown(st.session_state.guideline)
        st.stop()

    if col1.button("Get feedback"):
        report = st.session_state.feedback_chain.predict(
            input="Evaluate the interview.")
        st.markdown(report)
        st.download_button("Download feedback", report,
                           "resume_feedback.txt")
        st.stop()

    voice_mode = st.checkbox("Speak instead of typing")
    inp = (audio_recorder(pause_threshold=2.0, sample_rate=44_100)
           if voice_mode else st.chat_input("Your answer"))

    # wav_np is always defined (None until first answer)
    wav_np = handle_answer(inp) if inp else None

    for m in st.session_state.history:
        role = "assistant" if m.origin == "ai" else "user"
        with st.chat_message(role):
            st.write(m.message)
            if role == "assistant" and auto_play and wav_np is not None:
                st.audio(wav_np, sample_rate=24_000)
else:
    st.info("Select a position **and** upload a PDF resume to start.")
