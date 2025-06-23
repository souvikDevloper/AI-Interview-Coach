<h1 align="center">
  GPT Interviewer 🎙️
  <br>
  <sup><sub>Streamlit · LangChain · Fireworks · Edge-TTS · Whisper</sub></sup>
</h1>

<p align="center">
  <em>Your personal, on-device AI interviewer for résumé, JD &amp; behavioural practice.</em>
</p>

<p align="center">
  <img alt="GitHub Workflow Status" src="https://img.shields.io/github/actions/workflow/status/YourUser/GPTInterviewer/ci.yml?branch=main">
  <img alt="License" src="https://img.shields.io/github/license/YourUser/GPTInterviewer">
</p>

---

## Table of Contents
1. [Demo](#demo)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Roadmap](#roadmap)
8. [Contributing](#contributing)
9. [License](#license)

---

## Demo
> *Short GIF / YouTube link goes here.*  
> The app walks you through three professional interview modes—**Job-Description**, **Behavioural** and **Résumé**—with real-time speech and feedback.

---

## Key Features
| Mode | What it does | LLM ↔ Embedding |
|------|--------------|-----------------|
| **JD Screen** | Transforms any job-desc/keyword set into a topic guideline and grills you on technical depth. | `llama-v3p1-8b-instruct` ↔ `nomic-embed-text` |
| **Behavioural Screen** | Generates soft-skill questions, follows up on STAR answers, and critiques clarity & impact. | Same |
| **Résumé Screen** | Parses your PDF résumé, pinpoints achievements/skills, and asks position-specific questions. | Same |

*Edge-TTS voice*, *Whisper STT* and *instant feedback scoring* are available in every mode.

---

## Architecture

```mermaid
graph TD
  %% ────────── UI ─────────────
  subgraph Frontend (Streamlit)
    JD[Job Description textarea]
    MIC[Audio Recorder]
    CHAT[Chat Input / History]
  end

  %% ───── LangChain core ──────
  subgraph Backend (LangChain + Fireworks)
    RETR[FAISS + FireworksEmbeddings]
    GL[Guideline RetrievalQA]
    CONVO[ConversationChain (LLM)]
    STT[Whisper (STT)]
    TTS[Edge-TTS (TTS)]
  end

  %% ────────── flow ───────────
  JD  --> RETR
  RETR --> GL
  CHAT --> CONVO
  MIC  --> STT --> CONVO
  GL   --> CONVO
  CONVO --> TTS
  CONVO --> CHAT
Everything after the arrow heads is fully serverless—Fireworks handles the LLM & embedding calls, while Whisper & TTS run locally (CPU-only by default).

LangChain orchestration keeps prompt templates, vector search and memory consistent across screens.

Quick Start


# 1. Clone & enter the repo
git clone https://github.com/YourUser/GPTInterviewer.git
cd GPTInterviewer

# 2. Create a lightweight virtual-env
python -m venv .venv && .\.venv\Scripts\activate      # Windows
# source .venv/bin/activate                           # mac/Linux

# 3. Install runtime deps (no CUDA required)
pip install -r requirements.txt

# 4. Add your Fireworks key
cp .env.example .env
# edit .env and paste FIREWORKS_API_KEY=fw_****

# 5. Run!
streamlit run Homepage.py
Windows & Whisper note
We force CPU inference with CT2_FORCE_CPU=1 and install faster-whisper 1.0.2 + onnxruntime 1.22 to avoid missing cuDNN DLLs.

Configuration
Variable	Default	Description
FIREWORKS_API_KEY	–	Required. Grab a free key at https://fireworks.ai.
FIREWORKS_MODEL	accounts/fireworks/models/llama-v3p1-8b-instruct	Main chat model. Swap with any you see in GET /models.
EMBED_MODEL	nomic-embed-text	Text-embedding model for FAISS.
CT2_FORCE_CPU	1	Keep Whisper on CPU; set 0 if you have GPU + cuDNN.

Troubleshooting
Symptom	Fix
“ValueError: int8_float16 compute type not supported”	CT2_FORCE_CPU=1 or install a GPU build of cuDNN 9.
Large repo (>700 MB)	Run git filter-repo --invert-paths --path .venv --path models --path '*.bin' then force-push.
Mermaid diagram won’t render	Ensure every node label is inside exactly one pair of brackets and edges are on separate lines.

Roadmap
 Multi-user session storage (Redis)

 GitHub Action for auto-build & Streamlit Community Cloud deploy

 Additional Fireworks vision model for résumé screenshot analysis

 Custom LLM evaluation rubric for richer feedback

Contributing
Fork and create a feature branch.

pre-commit install to run Black, Ruff & isort.

Submit a PR—all contributors must sign off the DCO (Developer Certificate of Origin).

License
Licensed under the MIT License.
© 2025 Souvik Ghosh. All rights reserved.