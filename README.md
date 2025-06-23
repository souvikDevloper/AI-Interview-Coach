───────────────────────────────
README ▸ AI-Interview-Coach  ✨
────────────────────────────────────────────────────────────────── -->

<p align="center">
  <img src="docs/assets/logo.svg" width="160" alt="AI-Interview-Coach logo"/>
</p>

<h1 align="center">AI-Interview-Coach</h1>
<p align="center">
  <b>Your private recruiter-grade interviewer &nbsp;·&nbsp; runs 100 % locally or via Fireworks AI</b><br/>
  <img src="https://img.shields.io/badge/Streamlit-1.38-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/Fireworks%20AI-Serverless-ff69b4"/>
  <img src="https://img.shields.io/badge/Python-3.11-blue"/>
  <img src="https://img.shields.io/badge/License-MIT-green"/>
</p>

---

## ✨ What makes it shine?

| ✅ Feature | ✨ Highlights |
|------------|--------------|
| **Three interview modes** | *Professional* (tech deep-dive), *Behavioural* (soft-skills) & *Resume-based* grilling. |
| **Full voice loop** | 1-click recording → Whisper STT → LLM → Edge-TTS playback. |
| **Bring-your-own model** | Default to <br/>`llama-v3-8b-instruct` on **Fireworks AI**; drop-in OpenAI / Ollama support. |
| **Lightning retrieval** | Job-description chunks & resume PDFs embedded with **Fireworks-Embeddings** → **FAISS** similarity search. |
| **Guideline auto-drafting** | Each screen autogenerates a structured interview plan before the first question lands. |
| **Instant coaching** | “Get feedback” button rates the candidate, explains gaps & suggests targeted drills. |
| **100 % Streamlit** | Zero-JS front-end; deploy on 🐳 Docker, HuggingFace Spaces or Streamlit Cloud.

---

## 🏗️  High-level architecture

```mermaid
flowchart TD
    A[Microphone / Text box] -->|WAV| B(Whisper<br/>CPU ⇄ ONNX)
    B -->|Transcript| C{Interview<br/>Screen}
    subgraph Streamlit App
        C --> D[Retriever<br/>FAISS + FireworksEmbeddings]
        D -->|Relevant chunks| E[LLM<br/>Fireworks Chat]
        C -. guideline .-> E
        E -->|Question ↔ Answer| C
        E --> F[Edge-TTS]
    end
    F -->|WAV| G[HTML5 Audio]
The same core loop powers all three “screens”; only the retriever sources & prompt templates differ.

🗂️ Repository layout

GPTInterviewer/
├─ .venv/                   # isolated Python env  (optional)
├─ pages/
│  ├─ Professional Screen.py
│  ├─ Behavioural Screen.py
│  └─ Resume Screen.py
├─ prompts/
│  ├─ prompts.py            # Jinja-style templates
│  └─ prompt_selector.py
├─ speech_recognition/
│  └─ offline.py            # faster-whisper wrapper
├─ tts/
│  └─ edge_speak.py         # MS Edge neural voices
├─ images/                  # Lottie + UI art
└─ README.md
🚀 Quick-start (local ✨)

# 1 – clone & enter
git clone https://github.com/your-handle/AI-Interview-Coach.git
cd AI-Interview-Coach

# 2 – Python env
py -m venv .venv && .\.venv\Scripts\activate
pip install -r requirements.txt

# 3 – API keys
cp .env.template .env
#  └─ add FIREWORKS_API_KEY=fw_********************************

# 4 – run
streamlit run Homepage.py
CPU-only Windows build
Whisper & ONNX prefer GPU; on laptops set:


$Env:CT2_FORCE_CPU        = "1"
$Env:CUDA_VISIBLE_DEVICES = ""
…and install the lighter wheels:


pip install --no-binary :all: faster-whisper==1.0.2
pip install onnxruntime==1.22.0
🔧 Configuration
Variable	Default	Description
FIREWORKS_API_KEY	–	Serverless LLM & embedding calls
FIREWORKS_MODEL	accounts/fireworks/models/llama-v3p1-8b-instruct	Override with any model ID from GET /models
EMBED_MODEL	nomic-embed-text	Text embedding model for chunk vectors
EDGE_TTS_VOICE	en-US-AndrewNeural	Change narrator persona

📊 Scoring rubric (built-in)
Weight	Dimension
40 %	Technical accuracy & depth
30 %	Communication & clarity
20 %	Problem-solving approach
10 %	Poise (time-to-answer, filler words)

(Adjust in prompts/prompts.py if your org uses a different rubric.)

💡 Extending
Plug-in new models – swap ChatFireworks for ChatOpenAI or ChatOllama.

Add interview “screens” – copy a page, craft a template & a retriever source.

Fine-tune scoring – tweak the feedback prompt or chain in feedback_chain.

Dockerise – a Dockerfile is provided; just build & push to any container host.

🙌 Acknowledgements
Fireworks AI for generous inference credits

Whisper & faster-whisper community

Streamlit for the ridiculously productive UI

MIT-licensed – use it, fork it, improve it.
Pull requests & ⭐ stars are always welcome!