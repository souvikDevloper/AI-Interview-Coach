GPT Interviewer 🎙️
AI-powered mock interviews for tech roles — voice in, voice out, instant feedback.

<p align="center"> <img alt="Streamlit" src="https://img.shields.io/badge/Frontend-Streamlit-ff4c2e?logo=streamlit&logoColor=white"> <img alt="LangChain" src="https://img.shields.io/badge/Framework-LangChain-000"> <img alt="Fireworks" src="https://img.shields.io/badge/LLM-Fireworks_AI-ffcc00"> <img alt="Python" src="https://img.shields.io/badge/Python-3.11-blue?logo=python"> <img alt="License" src="https://img.shields.io/badge/License-MIT-green"> </p>
Table of contents
Highlights

Architecture

Prerequisites

Install & run

Environment config

Usage

Performance tips

Troubleshooting

Project layout

Contributing

License

Highlights
Three interview modes

Professional (JD-based) — paste a job description; the app builds a guideline and asks one focused technical question per topic.

Behavioral — STAR-style soft-skill questions with rubric-based feedback.

Résumé — parse your PDF and drill into your actual experience.

Voice I/O — local Whisper STT + Edge-TTS for natural conversation.

Retrieval-augmented — Fireworks embeddings → FAISS similarity search.

One-click report — strengths, gaps, and an overall score you can download.

Architecture

flowchart TD
  A[User] --> B[Streamlit UI]
  B --> C{Voice enabled?}
  C -- Yes --> D[AudioRecorder]
  D --> E[Whisper<br/>(faster-whisper)]
  C -- No --> F[Typed input]
  E --> F

  F --> G[LangChain Memory<br/>(ConversationBuffer)]
  F --> H[Retriever<br/>FAISS + FireworksEmbeddings]
  H --> I[Top-k Context]

  G --> J[ConversationChain Prompt]
  I --> J
  J --> K[ChatFireworks LLM<br/>llama-v3p1-8b-instruct]
  K --> L[Interviewer reply]

  L --> M{Auto-play voice?}
  M -- Yes --> N[Edge-TTS] --> O[Audio Out]
  M -- No --> B

  B -->|User clicks "Get feedback"| P[Feedback Chain]
  P --> Q[Scored Report .txt]
Prerequisites
OS: Windows 10/11 (works on macOS/Linux too)

Python: 3.11 recommended

Fireworks API key: create one in your Fireworks account

(Optional GPU for STT) HP Victus w/ NVIDIA RTX 3060
For GPU Whisper you’ll need:

NVIDIA driver (recent)

CUDA Toolkit 12.x or 13.x

cuDNN 9.x (ensure cudnn_ops64_9.dll is on PATH)

LLM calls run in Fireworks’ cloud; the GPU is only for local Whisper speech-to-text. CPU mode works fine if you don’t want the CUDA stack.

Install & run

# 1) clone
git clone https://github.com/<your-username>/GPTInterviewer.git
cd GPTInterviewer

# 2) venv
python -m venv .venv
.\.venv\Scripts\activate

# 3) deps (CPU-friendly by default)
pip install -r requirements.txt

# 4) env file
copy .env.example .env
# then open .env and paste your FIREWORKS_API_KEY

# 5) (OPTION A: CPU Whisper — safest)
setx CT2_FORCE_CPU "1"

# 6) run
streamlit run Homepage.py
(OPTION B) Enable GPU Whisper on RTX 3060 (faster, lower latency)
Install CUDA Toolkit (12.x or 13.x) and cuDNN 9.x.

Add both bin folders to your User PATH, then open a new PowerShell:


[Environment]::SetEnvironmentVariable(
  "PATH",
  $env:PATH + ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;C:\Program Files\NVIDIA\CUDNN\v9.12\bin\12.9",
  "User"
)
Sanity checks:


# ctranslate2 sees CUDA + supported types
$code = @"
import sys, ctranslate2
print('Python :', sys.version.split()[0])
print('CT2    :', ctranslate2.__version__)
print('CUDA devices        :', ctranslate2.get_cuda_device_count())
print('CUDA compute types  :', ctranslate2.get_supported_compute_types('cuda'))
"@
python -c $code

# quick Whisper GPU probe (float16)
$probe = @"
from faster_whisper import WhisperModel
WhisperModel('small', device='cuda', compute_type='float16')
print('✅ CUDA Whisper OK')
"@
python -c $probe
Keep CT2_FORCE_CPU unset (or set to 0) for GPU mode.

Environment config
Create .env (never commit this):


FIREWORKS_API_KEY=fw_xxxxxxxxxxxxxxxxxxxxx
FIREWORKS_MODEL=accounts/fireworks/models/llama-v3p1-8b-instruct
EMBED_MODEL=nomic-embed-text
You can swap to any model visible to your key (run GET /models or use the PowerShell snippet you tested earlier).

Usage
Homepage: choose Professional, Behavioral, or Résumé.

Professional: paste a JD → click Start Interview → answer in chat or voice.

Behavioral: supply prompt/keywords → one question at a time, STAR-guided.

Résumé: upload PDF → questions align with your actual experience.

Feedback: click Get feedback any time for a scored, downloadable report.

The app purposely asks one question at a time and uses memory so it won’t repeat itself. You can set a max-turn limit (see Performance tips).

Performance tips
GPU Whisper: if you enabled CUDA, STT latency drops a lot on RTX 3060.

Chunking: in the pages we use NLTKTextSplitter(); for long JDs/resumés you can tune chunk size/overlap to reduce token load and speed up RAG.

Retriever: keep k=3–5 similar chunks. Higher values increase context size.

Cap turns: each screen has MAX_TURNS to keep interviews ~10–15 minutes.

Voice off: if your mic is flaky, disable voice to remove STT overhead.

Troubleshooting
Symptom	Fix
cudnn_ops64_9.dll missing / int8_float16 error	CPU Whisper: setx CT2_FORCE_CPU 1 and restart terminal. Or complete CUDA+cuDNN path setup and use compute_type='float16'.
Fireworks NOT_FOUND / 404	Use the full model slug from GET /models, e.g. accounts/fireworks/models/llama-v3p1-8b-instruct.
“Prompt is too long”	Your JD/résumé is huge. Reduce chunk size / overlap or lower retriever k.
ModuleNotFoundError: streamlit_lottie or others	pip install -r requirements.txt inside the activated .venv.
“get_pages not found”	We avoid internal APIs — homepage uses native navigation now. No action needed if you’re on the current code.
Repo too big on push	You accidentally committed .venv/ or models/**. Add to .gitignore, then clean history with git filter-repo --invert-paths --path .venv --path models --path '*.bin' --path '*.dll' --path '*.pyd'.

Project layout

GPTInterviewer/
├─ Homepage.py                 # entry screen (no internal hacks)
├─ pages/
│  ├─ Professional Screen.py   # JD-driven interview
│  ├─ Behavioral Screen.py     # soft-skill interview
│  └─ Resume Screen.py         # resume-driven interview
├─ prompts/
│  ├─ prompts.py               # templates
│  └─ prompt_selector.py       # per-position selectors
├─ speech_recognition/
│  └─ offline.py               # faster-whisper wrapper (CPU/GPU)
├─ tts/
│  └─ edge_speak.py            # Edge-TTS helper
├─ app_utils.py                # NLTK bootstrap, misc helpers
├─ requirements.txt
├─ .env.example
└─ .gitignore
Contributing
This is a personal project. Issues and suggestions are welcome.
Pull requests: feel free to open a PR on a feature branch, but final merges are at the discretion of the repository owner to keep scope and quality tight.

License
MIT © 2025 Souvik Ghosh

Known-good versions (pinned)
python 3.11.x

streamlit 1.33+

langchain 0.2.x / langchain-core 0.2.x

langchain-fireworks latest compatible with above

faster-whisper 1.0.2 (CPU-safe)

onnxruntime 1.22.0 (CPU)

ctranslate2 4.6.0

faiss-cpu latest