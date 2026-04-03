# Deployment Document (Windows-First, Free Hosting)

## 1. Why This Document Exists

This guide explains how to deploy the full DeepFake Defender system from a Windows machine to free hosting services.

Main goals:

1. Keep the current working behavior unchanged.
2. Deploy with clear and repeatable steps.
3. Avoid common failures related to Python version, dependencies, model files, and stale process output.

## 2. Important Reality About Free Hosting

You asked for deployment without lag. For free services, there is one hard limit:

1. Most free services sleep when idle.
2. The first request after sleep can be slower (cold start).

What this means:

1. You can get good performance after warm-up.
2. You cannot fully remove cold-start delay on free plans.

How to reduce delay is covered in Section 10.

## 3. Recommended Free Deployment Options

### Option A (Recommended): Single Service on Render

Use one backend service that serves both:

1. Flask pages (image and video UI)
2. API endpoints

Why this is easiest:

1. No CORS setup needed.
2. No frontend-backend domain sync issues.
3. Lowest moving parts.

### Option B: Split Deployment (Vercel + Render)

1. React frontend on Vercel (free)
2. Flask backend on Render (free)

Use this only if you need React hosting separately.

## 4. Pre-Deployment Checklist (Windows)

Run from repo root:

```powershell
python --version
node --version
npm --version
```

Recommended versions:

1. Python 3.10 to 3.13 (project tested on 3.13.3)
2. Node 18+

Set up local environment and verify first:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Run backend health check locally:

```powershell
Set-Location .\backend
& "..\.venv\Scripts\python.exe" app.py
```

In another terminal:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:7860/api/health
```

Expected:

1. success: true
2. c2pa.available: true
3. c2pa.has_reader: true

## 5. Model File Readiness (Critical)

Before cloud deployment, confirm required files are available in the deployment environment.

Required image files:

1. model_output/resnet50_finetuned_benchmark.pth
2. ai_detector_meta_learner.joblib
3. polynomial_transformer.joblib
4. model_output/synthid/robust_codebook.pkl

Optional video checkpoints:

1. backend/checkpoints/efficientnet.onnx
2. backend/checkpoints/model.pth

If optional video files are missing, fallback video mode still works.

Important note about Git ignore:

1. This repository ignores many large model file patterns.
2. If files are not in your remote repo, deployment will fail or partially degrade.
3. Make sure model assets are available in cloud runtime (tracked files, artifact download step, or mounted storage).

## 6. Option A Step-by-Step: Deploy Full App to Render (Free)

### 6.1 Push code to GitHub

From Windows PowerShell:

```powershell
Set-Location "E:\AI Media Detection\AI-media-detection-tool"
git add .
git commit -m "Prepare deployment"
git push
```

### 6.2 Create Render service

1. Sign in to Render.
2. Click New > Web Service.
3. Connect your GitHub repo.
4. Select branch.

Use settings:

1. Environment: Python
2. Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
3. Start Command: `python backend/serve.py`
4. Health Check Path: `/api/health`

Render automatically provides PORT. The app reads it.

### 6.3 Deploy and verify

After deploy finishes:

1. Open `https://<your-render-url>/api/health`
2. Open home page `/`
3. Run one image scan and one video scan

### 6.4 If C2PA is unavailable on cloud

1. Check `/api/health` output first.
2. If c2pa.available is false, review deploy logs for c2pa import error.
3. Keep service running with SynthID + AI model while investigating C2PA runtime package support.

## 7. Option B Step-by-Step: Vercel Frontend + Render Backend (Free)

Use this when you want a standalone React deployment.

### 7.1 Deploy backend to Render first

Use Section 6 to publish backend and get URL.

### 7.2 Deploy frontend to Vercel

1. Import the same repo in Vercel.
2. Set project root to `frontend`.
3. Build command: `npm run build`
4. Output directory: `dist`

### 7.3 Route frontend API traffic to backend

Because frontend uses `/api/...` paths, configure rewrite/proxy in Vercel so `/api/*` goes to your Render backend.

If rewrite is not set, frontend will fail API calls.

## 8. Post-Deployment Validation Checklist

Run these checks after every deployment:

1. GET `/api/health` returns success true.
2. GET `/` loads UI.
3. POST `/api/analyze` works for one sample image.
4. POST `/api/analyze_video` works for one sample video.
5. POST `/api/forensic-report` returns report text.
6. C2PA policy text appears correctly when C2PA metadata is present.

## 9. Common Deployment Problems and Fixes

### Problem 1: Output looks old after new deployment

Fix:

1. Confirm request is hitting new URL.
2. Clear browser cache.
3. Run a fresh scan (not old sessionStorage data).
4. Verify backend commit hash/logs on host.

### Problem 2: App starts but model inference fails

Fix:

1. Check model file presence in cloud runtime.
2. Confirm file paths match project expectations.
3. Check logs for missing file messages.

### Problem 3: C2PA check says unavailable

Fix:

1. Check `/api/health` c2pa fields.
2. Review deploy logs for import error.
3. Ensure `c2pa-python` was installed from `requirements.txt`.

### Problem 4: Frontend calls fail in split setup

Fix:

1. Confirm backend URL is reachable.
2. Confirm frontend rewrite for `/api/*` is set.
3. Test a direct backend endpoint in browser.

### Problem 5: Slow first request

Fix:

1. This is normal on free sleeping services.
2. Use warm-up ping strategy (Section 10).

## 10. How to Reduce Lag on Free Plans

You cannot fully remove lag on free plans, but you can reduce it:

1. Keep one service (Option A) instead of split services.
2. Keep model loading at startup (already done in this project).
3. Use smallest practical worker count to save memory.
4. Use a free uptime ping service to call `/api/health` every few minutes.
5. Keep uploads reasonable in size.
6. Keep logs clean to spot performance drops early.

## 11. Security Checklist for Public Deployment

1. Use HTTPS only.
2. Keep debug mode off.
3. Do not expose internal file paths in user-facing errors.
4. Keep dependency updates controlled and tested.
5. Limit upload size and accepted file extensions.
6. Monitor logs for repeated bad requests.

## 12. Safe Update Procedure

When pushing new code:

1. Deploy to same service.
2. Wait for deploy success.
3. Run validation checklist in Section 8.
4. If failure appears, rollback to last working commit.

Simple rollback command from Windows:

```powershell
git log --oneline
git revert <bad-commit-hash>
git push
```

## 13. Quick Local Commands (Windows)

Start backend dev:

```powershell
Set-Location .\scripts\windows
.\start-backend-dev.ps1
```

Start backend production mode:

```powershell
Set-Location .\scripts\windows
.\start-backend-prod.ps1
```

Start frontend dev:

```powershell
Set-Location .\scripts\windows
.\start-frontend-dev.ps1
```

Check health:

```powershell
Set-Location .\scripts\windows
.\health-check.ps1
```

## 14. Final Recommendation

For easiest and most stable free deployment:

1. Use Option A (single Render service).
2. Keep backend and UI together.
3. Validate with `/api/health` after every deploy.
4. Keep model files available in cloud runtime at all times.
