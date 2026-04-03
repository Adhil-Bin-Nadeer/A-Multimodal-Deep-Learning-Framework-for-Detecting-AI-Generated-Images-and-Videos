# DeepFake Defender (AI Media Detection Tool)

DeepFake Defender checks images and videos using multiple layers instead of one model.
This gives more stable results and reduces conflicting output.

This README is written for Windows users first, and it explains setup, versions, dependencies, troubleshooting, and safe upgrade steps in detail.

## 1. What This Project Does

For image detection, the backend uses this order:

1. C2PA signed metadata check.
2. SynthID watermark check.
3. AI model plus forensic checks (only if needed).

Current policy in this project:

1. If C2PA metadata is present, the file is treated as AI-generated and lower layers are skipped.
2. If C2PA is not present but SynthID watermark is detected, AI model layer is skipped.
3. If both above are not decisive, model plus forensic layer decides.

For video detection, the system:

1. Tries dedicated video checkpoints when available.
2. Falls back to frame-based scoring if dedicated checkpoints are missing.

## 2. Current UI Surfaces

The repository has two UI options:

1. Flask templates in [backend/templates](backend/templates) and static assets in [backend/static](backend/static).
2. React app in [frontend](frontend) with routes for image and video scan and report flows.

## 3. Tested Environment and Version Guidance

Recommended and tested:

1. OS: Windows 10 or 11.
2. Python: 3.13.3 in a virtual environment.
3. Node.js: 18 or newer.
4. npm: 9 or newer.

Supported Python range for practical use:

1. Python 3.10 to 3.13.

Important:

1. This repo uses one dependency file: [requirements.txt](requirements.txt).
2. Do not create a second requirements file unless you fully maintain both.
3. Always run backend using the virtual environment interpreter path to avoid mixed package environments.

## 4. Repository Structure (Important Paths)

1. [backend/app.py](backend/app.py): Main Flask app and API routes.
2. [backend/serve.py](backend/serve.py): Production entrypoint (Waitress).
3. [backend/combine_model.py](backend/combine_model.py): Image model fusion and thresholds.
4. [backend/custom_forensics.py](backend/custom_forensics.py): Frequency, anatomy, and entropy checks.
5. [backend/forensic.py](backend/forensic.py): Human-readable forensic summaries.
6. [backend/src/c2pa_checker.py](backend/src/c2pa_checker.py): C2PA runtime and manifest checks.
7. [backend/video_detect_standalone.py](backend/video_detect_standalone.py): Video path and fallback.
8. [frontend/src/pages](frontend/src/pages): React pages (image and video flows).
9. [scripts/windows](scripts/windows): Helper scripts for Windows startup and health checks.
10. [model_output](model_output): Model artifacts required by runtime.

## 5. Required Runtime Assets

Required for image AI path:

1. [model_output/resnet50_finetuned_benchmark.pth](model_output/resnet50_finetuned_benchmark.pth)
2. [ai_detector_meta_learner.joblib](ai_detector_meta_learner.joblib)
3. [polynomial_transformer.joblib](polynomial_transformer.joblib)

Required for SynthID:

1. [model_output/synthid/robust_codebook.pkl](model_output/synthid/robust_codebook.pkl)

Optional for dedicated video path:

1. [backend/checkpoints/efficientnet.onnx](backend/checkpoints/efficientnet.onnx)
2. [backend/checkpoints/model.pth](backend/checkpoints/model.pth)

If optional video checkpoints are missing, video still works with fallback scoring.

## 6. Full Windows Setup (From Scratch)

From repository root:

1. Create virtual environment:

```powershell
python -m venv .venv
```

2. Activate virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Upgrade pip tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

4. Install backend dependencies:

```powershell
python -m pip install -r requirements.txt
```

5. Install frontend dependencies:

```powershell
Set-Location .\frontend
npm install
Set-Location ..
```

## 7. Run the Project (Windows)

### Option A: Use helper scripts (recommended)

Start backend in development mode:

```powershell
Set-Location .\scripts\windows
.\start-backend-dev.ps1
```

Start frontend in another terminal:

```powershell
Set-Location .\scripts\windows
.\start-frontend-dev.ps1
```

Run backend health check:

```powershell
Set-Location .\scripts\windows
.\health-check.ps1
```

### Option B: Manual commands

Backend (development mode):

```powershell
Set-Location .\backend
& "..\.venv\Scripts\python.exe" app.py
```

Frontend (development mode):

```powershell
Set-Location .\frontend
npm run dev
```

Vite development proxy is configured to backend port 7860 in [frontend/vite.config.js](frontend/vite.config.js).

## 8. Production Mode (Local or Server)

Use Waitress (not Flask development server):

```powershell
Set-Location .\scripts\windows
.\start-backend-prod.ps1
```

Or manual:

```powershell
Set-Location .\backend
& "..\.venv\Scripts\python.exe" serve.py
```

Environment variables you can set:

1. PORT (default: 7860)
2. HOST (default: 0.0.0.0)
3. WAITRESS_THREADS (default: 8)
4. ENABLE_TTA (default: 1)
5. RESNET_AI_INDEX (default: 1)
6. VIT_AI_INDEX (used only as fallback)

## 9. API and Health Endpoints

Main endpoints:

1. GET /api/health
2. POST /api/analyze
3. POST /api/analyze_video
4. POST /api/forensic-report

Health endpoint sample check:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:7860/api/health
```

What to verify in health response:

1. python_executable should point to your .venv path.
2. c2pa.available should be true.
3. c2pa.has_reader should be true.
4. models.ai_predictor_loaded should be true.
5. models.synthid_loaded should be true.

## 10. C2PA Troubleshooting (Most Common Cases)

Issue: C2PA says unavailable.

1. Confirm backend was started with virtual environment interpreter path.
2. Run health endpoint and inspect c2pa fields.
3. Reinstall package in active virtual environment:

```powershell
python -m pip install --upgrade c2pa-python
```

Issue: You still see old forensic text after code update.

1. Stop stale backend process using port 7860.
2. Restart backend with virtual environment interpreter.
3. Start a new scan (report pages use sessionStorage, so old results may still display until a new scan).

Issue: C2PA returns No C2PA manifest found.

1. This is not a library failure.
2. It means the uploaded file has no signed C2PA metadata.

## 11. Frontend Troubleshooting

Issue: Video card still says Coming Soon.

1. Make sure you run the updated React build or development server.
2. Clear browser cache and reload.
3. Confirm route entries exist in [frontend/src/App.jsx](frontend/src/App.jsx).

Issue: Frontend works but API calls fail.

1. Check backend is running on port 7860.
2. Verify proxy target in [frontend/vite.config.js](frontend/vite.config.js).
3. Check browser dev tools Network tab for failed /api requests.

## 12. Dependency Management and Safe Upgrade Process

When Python version changes or system is moved:

1. Delete old virtual environment folder.
2. Recreate virtual environment with new Python.
3. Reinstall dependencies from [requirements.txt](requirements.txt).
4. Run health check.
5. Run one image scan and one video scan as smoke tests.

Suggested smoke tests after upgrades:

1. /api/health returns success true.
2. /api/analyze returns layers and final_verdict.
3. /api/analyze_video returns label and confidence.
4. /api/forensic-report returns enhanced_report.

## 13. Security and Stability Notes

1. Keep Flask debug mode off in production.
2. Use Waitress for production serving.
3. Keep upload size limits under control.
4. Do not log sensitive user file paths publicly.
5. Keep model files outside public static folders.
6. Use HTTPS when deploying publicly.

## 14. Performance Notes

1. First startup is heavier because models are loaded once.
2. Free hosting can introduce cold starts and slower response.
3. Keep model files on fast storage.
4. Avoid running multiple stale backend processes.

## 15. Deployment Guide Reference

Detailed free-hosting deployment plans for Windows users are in:

1. [deployment_document.md](deployment_document.md)

## 16. Technical Study Reference

Complete technical deep study in LaTeX is in:

1. [repository_study.tex](repository_study.tex)

## 17. License

This project is licensed under MIT. See [LICENSE](LICENSE).
