<h1 align="center">
  GPT&nbsp;Interviewer 🎙️
</h1>
<p align="center">
  <i>“Practice like it’s the real thing.”</i><br>
  AI-powered mock-interviews for tech candidates – voice in, voice out, feedback in seconds.
</p>

<p align="center">
  <img alt="Streamlit" src="https://img.shields.io/badge/Frontend-Streamlit-ff4c2e?logo=streamlit&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/Framework-LangChain-000?logo=data&logoColor=white">
  <img alt="Fireworks" src="https://img.shields.io/badge/LLM-Fireworks_AI-ffcc00">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?logo=python">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## ✨ Highlights

| 🚀 Feature | 💡 What you get |
|-----------|----------------|
| **JD Screen** | Paste a job-description → app auto-builds a guideline & asks *one* targeted tech question per topic. |
| **Behavioural Screen** | STAR-style soft-skill questions, intelligent follow-ups, instant rubric feedback. |
| **Résumé Screen** | Parses your PDF, spots skills & gaps, grills you accordingly. |
| **Voice I/O** | Local Whisper STT + Microsoft Edge-TTS = natural conversation. |
| **Downloadable Report** | Score + strengths + improvement tips in a single click. |

---

## 🗄️ Project Structure (core parts)

GPTInterviewer/
│
├─ pages/ # Streamlit multi-page UI
│ ├── Homepage.py
│ ├── Professional Screen.py
│ ├── Behavioural Screen.py
│ └── Resume Screen.py
│
├─ speech_recognition/ # Whisper (CPU-only) helper
├─ tts/ # Edge-TTS wrapper
├─ prompts/ # Prompt templates + selector
├─ requirements.txt
└─ .env.example # add your Fireworks key here



### Under the hood 🔧
1. **Retriever** – Fireworks embeddings ➜ FAISS similarity search  
2. **Guideline builder** – `RetrievalQA` (LangChain) + `llama-v3p1-8b-instruct`  
3. **Conversation** – `ConversationChain` keeps context & memory  
4. **Audio** (opt-in) – Whisper (STT) and Edge-TTS (synth)  
5. **Feedback** – second LLM pass with a scoring prompt

Everything except audio runs in the free Fireworks cloud; Whisper/Edge-TTS stay local (CPU-only by default).

---

## ⚡ Quick Start


git clone https://github.com/souvikDevloper/GPTInterviewer.git
cd GPTInterviewer

python -m venv .venv && ^        # Win
.\.venv\Scripts\activate         # Win
# source .venv/bin/activate      # mac/Linux

pip install -r requirements.txt

# --- configure -------------------------------------------------
copy .env.example .env           # Win
# cp .env.example .env           # mac/Linux
# open .env and paste FIREWORKS_API_KEY=fw_xxxxxxxxx

# (optional) keep Whisper on CPU – safe on any box
set CT2_FORCE_CPU=1              # Win
# export CT2_FORCE_CPU=1         # mac/Linux
# ----------------------------------------------------------------

streamlit run Homepage.py
Tip: No GPU? You’re fine – the repo pins faster-whisper==1.0.2 + onnxruntime==1.22.0 (CPU wheels).

🔧 Key Settings
Env var	Default	Notes
FIREWORKS_API_KEY	–	Required – create one at https://fireworks.ai.
FIREWORKS_MODEL	accounts/fireworks/models/llama-v3p1-8b-instruct	Swap to any public model you prefer.
EMBED_MODEL	nomic-embed-text	Remote embedding model for retriever.
CT2_FORCE_CPU	1	Whisper CPU fallback. Set to 0 if you have CUDA + cuDNN 9.

🩹 Common Pitfalls
Problem	Fix
cudnn_ops64_9.dll missing / int8_float16 error	Keep Whisper on CPU → set CT2_FORCE_CPU=1, reinstall faster-whisper==1.0.2.
Fireworks 404 / “model not found”	Use the full model slug from GET /models, e.g. accounts/fireworks/models/llama-v3p1-8b-instruct.
Repo > 100 MB	You pushed .venv/ or models/**. Delete local venv, commit .gitignore, then run:
git filter-repo --invert-paths --path .venv --path models --path '*.bin'

🚀 Roadmap
Redis session storage (multi-user)

CI workflow → Streamlit Community deploy

Vision LLM: résumé screenshot critique

Customisable scoring weights

© License & Credits
MIT License – 2025 Souvik Ghosh (solo author).
Built with Streamlit, LangChain, Fireworks AI, Whisper, Edge-TTS & FAISS.

Pull-requests are welcome (feature branches only) – but final merges remain at the discretion of the repository owner.