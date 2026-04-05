# DeepFake Defender (AI Media Detection Tool)

DeepFake Defender is a layered image and video authenticity analyzer.
It does not rely on one model. It combines provenance checks, watermark checks, deep learning models, and forensic signals.

This README is for new users cloning the project for the first time, especially on Windows where environment and runtime issues are common.

## Table of Contents

1. Project Overview
2. Detection Policy and Decision Order
3. Repository Layout
4. System Requirements
5. Clone and First-Time Setup
6. Runtime Assets (Required vs Optional)
7. Run the Project (Development and Production)
8. Health Check and Readiness
9. API Endpoints and Response Contracts
10. Environment Variables
11. Windows-Specific Issues and Fixes
12. Cross-Platform Notes (macOS/Linux)
13. Result Interpretation Notes
14. Security and Deployment Notes
15. Technical Study Reference
16. License

## 1. Project Overview

The project has two UI surfaces and one backend:

1. Flask template UI in `backend/templates` and `backend/static`.
2. React UI in `frontend` (Vite + React Router).
3. Flask backend API in `backend/app.py`.

Main capabilities:

1. Image analysis through three-layer cascade:
- C2PA signed metadata
- SynthID watermark
- ResNet + ViT + forensics
2. Video analysis:
- Dedicated checkpoint path when video checkpoints exist
- Fallback frame-based path when checkpoints are missing

## 2. Detection Policy and Decision Order

Image decision order is strict and short-circuited:

1. Layer 1 (C2PA): if C2PA metadata is present, final verdict is AI-generated and lower layers are skipped.
2. Layer 2 (SynthID): if C2PA is absent and calibrated SynthID watermark is detected, final verdict is AI-generated and model layer is skipped.
3. Layer 3 (AI model + forensics): runs only when Layers 1 and 2 are not decisive.

Important policy behavior currently in code:

1. C2PA metadata presence alone triggers AI decision (`AI Generated (C2PA Metadata Present)`) even without explicit AI assertion.
2. SynthID has calibrated thresholds; weak raw signals do not automatically force AI verdict.
3. Model layer uses dynamic thresholds that depend on forensic support and disagreement between ResNet and ViT.

## 3. Repository Layout

Top-level important files and folders:

- `backend/app.py`: Flask app, routes, layer orchestration, runtime initialization.
- `backend/combine_model.py`: image model fusion, threshold logic, guards, confidence computation.
- `backend/custom_forensics.py`: frequency/anatomy/entropy analyzers and custom score calibration.
- `backend/forensic.py`: user-facing forensic summary and report generation.
- `backend/video_detect_standalone.py`: video detection (dedicated and fallback).
- `backend/src/c2pa_checker.py`: C2PA runtime and manifest parsing.
- `backend/src/synthid/service.py`: SynthID service wrapper and calibration logic.
- `scripts/windows`: Windows startup and health-check scripts.
- `model_output`: model artifacts used at runtime.
- `frontend/src/pages`: React pages for image and video workflows.
- `repository_study.tex`: full technical handbook.

## 4. System Requirements

### Minimum

1. OS:
- Windows 10/11 (first-class support in scripts)
- macOS/Linux (manual commands supported)
2. Python: 3.10 to 3.13
3. Node.js: 18+
4. npm: 9+
5. RAM: 8 GB recommended
6. Disk: 3+ GB free (models + dependencies)

### Optional but useful

1. CUDA GPU for faster inference
2. Git LFS for model artifact retrieval
3. Stable internet for first startup model downloads (if local artifacts are missing)

## 5. Clone and First-Time Setup

## 5.1 Clone

```powershell
git clone <your-repo-url>
cd AI-media-detection-tool
```

## 5.2 Git LFS (recommended)

Model artifacts are tracked with LFS patterns (`.gitattributes` includes `*.pth`, `*.pkl`, `*.joblib`).

```powershell
git lfs install
git lfs pull
```

If you skip LFS or artifacts are missing, backend startup also attempts Hugging Face download for key files.

## 5.3 Create and activate Python virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If script execution is blocked on Windows:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 5.4 Install backend dependencies

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 5.5 Install frontend dependencies

```powershell
Set-Location .\frontend
npm install
Set-Location ..
```

## 5.6 Verify key runtime files exist

Expected key files:

- `model_output/resnet50_finetuned_benchmark.pth`
- `ai_detector_meta_learner.joblib`
- `polynomial_transformer.joblib`
- `model_output/synthid/robust_codebook.pkl`

If missing, backend may download them from Hugging Face on startup (`Adhil786/deepfake-models`) when internet is available.

## 6. Runtime Assets (Required vs Optional)

| Asset | Path | Required | If Missing |
|---|---|---|---|
| ResNet checkpoint | `model_output/resnet50_finetuned_benchmark.pth` | Yes | Image model cannot initialize |
| Meta learner | `ai_detector_meta_learner.joblib` | No | Meta fusion disabled; baseline blend used |
| Polynomial transformer | `polynomial_transformer.joblib` | No | Meta fusion disabled; baseline blend used |
| SynthID codebook | `model_output/synthid/robust_codebook.pkl` | No (for full pipeline yes) | SynthID layer unavailable; pipeline continues to model |
| ViT weights | Hugging Face (`dima806/ai_vs_real_image_detection`) | No | Uses ResNet + forensics only |
| Video ONNX checkpoint | `backend/checkpoints/efficientnet.onnx` | No | Video falls back to frame-based path |
| Video PyTorch checkpoint | `backend/checkpoints/model.pth` | No | Video falls back to frame-based path |
| MediaPipe task files | `model_output/mediapipe/*.task` or env paths | No | Anatomy forensic module can be unavailable |

## 7. Run the Project

### 7.1 Windows development (recommended)

Start backend:

```powershell
Set-Location .\scripts\windows
.\start-backend-dev.ps1
```

Start frontend in a second terminal:

```powershell
Set-Location .\scripts\windows
.\start-frontend-dev.ps1
```

Default ports:

1. Backend: 7860
2. Frontend (Vite): 3000

Vite proxy in `frontend/vite.config.js` forwards `/api/*` to backend port 7860.

### 7.2 Windows production-style backend

```powershell
Set-Location .\scripts\windows
.\start-backend-prod.ps1 -Port 7860 -Threads 8
```

This runs `backend/serve.py` with Waitress.

### 7.3 Manual run (any platform)

Backend:

```bash
cd backend
../.venv/Scripts/python.exe app.py   # Windows
# or
../.venv/bin/python app.py           # macOS/Linux
```

Frontend:

```bash
cd frontend
npm run dev
```

## 8. Health Check and Readiness

Health endpoint:

- `GET /api/health`

Windows helper script:

```powershell
Set-Location .\scripts\windows
.\health-check.ps1
```

Verify these fields before scanning:

1. `python_executable` points to project `.venv`.
2. `models.ai_predictor_loaded = true`.
3. `models.synthid_loaded = true` (or understand fallback behavior).
4. `runtime_initialized = true`.

If `runtime_initialized = false`, image endpoint can return 503 while background model load is still in progress.

## 9. API Endpoints and Response Contracts

## 9.1 `GET /api/health`

Returns runtime state, Python path/version, model readiness, and C2PA runtime status.

## 9.2 `POST /api/analyze`

Input:

- Multipart form-data with file key `file`
- Allowed image extensions: `png`, `jpg`, `jpeg`, `webp`
- Max request size: 16 MB

Top-level response keys:

- `success`
- `filename`
- `layers` (`c2pa`, `synthid`, `ai_model`)
- `final_verdict`
- `confidence`
- `is_ai_generated`
- `forensic_summary`
- `explainability`

Layer status values you may see:

- C2PA: `verified`, `present`, `not_found`, `unavailable`
- SynthID: `complete`, `skipped`, `unavailable`, `error`
- AI model: `complete`, `skipped`, `error`

## 9.3 `POST /api/analyze_video`

Input:

- Multipart form-data with file key `file`
- Allowed extensions: `mp4`, `avi`, `mov`

Returns:

- `label` (`FAKE` or `REAL`)
- `confidence`
- `ai_probability`
- `source`
- `explainability`
- `forensic_summary`

## 9.4 `POST /api/forensic-report`

Input:

- JSON analysis result from image flow

Returns:

- `enhanced_report`
- `summary`
- `forensic_summary`

## 10. Environment Variables

### App/runtime

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `7860` | Backend port |
| `HF_TOKEN` | unset | Hugging Face token (if needed for model downloads) |

### Model fusion (`backend/combine_model.py`)

| Variable | Default | Meaning |
|---|---|---|
| `ENABLE_TTA` | `1` | Enables mirrored-image TTA scoring |
| `RESNET_AI_INDEX` | `1` | ResNet class index treated as AI |
| `SKIP_VIT` | `0` | Skip ViT loading when set to `1` |
| `VIT_AI_INDEX` | `1` | Fallback ViT AI index if labels cannot be inferred |

### MediaPipe forensic module (`backend/custom_forensics.py`)

| Variable | Default | Meaning |
|---|---|---|
| `MEDIAPIPE_POSE_MODEL_PATH` | unset | Path override for `pose_landmarker.task` |
| `MEDIAPIPE_HAND_MODEL_PATH` | unset | Path override for `hand_landmarker.task` |

### Waitress (`backend/serve.py`)

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `7860` | Bind port |
| `WAITRESS_THREADS` | `8` | Worker thread count |

### SynthID calibration (`backend/src/synthid/service.py`)

| Variable | Default |
|---|---|
| `SYNTHID_MIN_CONFIDENCE` | `0.58` |
| `SYNTHID_MIN_PHASE_MATCH` | `0.48` |
| `SYNTHID_MIN_CORRELATION_MARGIN` | `0.0018` |
| `SYNTHID_MIN_CARRIER_MATCH_RATIO` | `0.30` |
| `SYNTHID_MIN_SIGNAL_SCORE` | `0.56` |
| `SYNTHID_MIN_STRUCTURE_RATIO` | `0.78` |
| `SYNTHID_MAX_STRUCTURE_RATIO` | `1.85` |
| `SYNTHID_MAX_MULTI_SCALE_STD` | `0.18` |

## 11. Windows-Specific Issues and Fixes

## 11.1 Wrong Python interpreter is running backend

Symptom:

- `/api/health` shows system Python instead of project `.venv`.
- Features differ from expected (for example C2PA unavailable unexpectedly).

Fix:

1. Stop backend.
2. Restart with `scripts/windows/start-backend-dev.ps1`.
3. Re-check `/api/health`.

## 11.2 Port 7860 already in use

Symptom:

- Backend fails to bind or old behavior persists.

Fix:

```powershell
netstat -ano | findstr :7860
taskkill /PID <PID> /F
```

Then restart backend.

## 11.3 Models still initializing (503 from `/api/analyze`)

Symptom:

- API returns runtime not initialized.

Fix:

1. Wait for background load completion.
2. Check `/api/health` until `runtime_initialized=true`.

## 11.4 ViT download is slow or blocked

Symptom:

- Startup delay or ViT load warnings.

Fix:

- Run with `SKIP_VIT=1` to use ResNet + forensics mode.

## 11.5 Anatomy module unavailable (MediaPipe Tasks API)

Symptom:

- Forensic output says anatomy check unavailable.
- Threshold can become stricter in low-corroboration cases.

Fix:

1. Install/repair `mediapipe`.
2. Provide task files:
- `pose_landmarker.task`
- `hand_landmarker.task`
3. Place in `model_output/mediapipe/` or set environment variable paths.

## 11.6 C2PA unavailable

Symptom:

- Health reports C2PA unavailable.

Fix:

```powershell
python -m pip install --upgrade c2pa-python
```

Recheck health.

## 11.7 Frontend shows old result

Symptom:

- Report page appears stale.

Reason:

- Result pages rely on `sessionStorage` from latest scan.

Fix:

- Run a fresh scan from dashboard.

## 12. Cross-Platform Notes (macOS/Linux)

The codebase is Windows-first in helper scripts, but backend/frontend work on macOS/Linux with manual commands.

Typical flow:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cd frontend && npm install
```

Run backend manually from `backend` and frontend with `npm run dev`.

## 13. Result Interpretation Notes

1. The displayed `confidence` is decision confidence relative to threshold margin, not the raw AI probability.
2. `ai_probability` can be high while final verdict is real if dynamic threshold is higher (for example low-corroboration guard case).
3. Forensic summaries include only available modules. Unavailable modules may be omitted from summary text.

## 14. Security and Deployment Notes

1. Development server (`app.py`) is for local development only.
2. Use Waitress (`serve.py`) for production-style serving.
3. Keep uploaded files temporary and private. Backend already deletes temp uploads after processing.
4. Use HTTPS and standard reverse proxy controls when deploying publicly.

For deployment guidance, see `deployment_document.md`.

## 15. Technical Study Reference

For deep technical flow, equations, thresholds, API semantics, and architecture details, read:

- `repository_study.tex`

## 16. License

MIT License. See `LICENSE`.
