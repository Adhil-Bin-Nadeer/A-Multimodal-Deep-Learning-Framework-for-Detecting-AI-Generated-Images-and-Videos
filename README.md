# AI Media Detection Tool (DeepFake Defender)

DeepFake Defender is a comprehensive tool designed to detect AI-generated media and verify image authenticity using multiple analysis layers, including C2PA metadata checking and ensemble AI models.

## 🚀 Features

- **Multi-Layer Detection**:
  - **C2PA Verification**: Checks for Content Provenance and Authenticity metadata.
  - **AI Ensemble Models**: Uses deep learning models (PyTorch, Transformers) to identify synthetic patterns.
  - **Forensic Reporting**: Generates detailed forensic analysis reports for uploaded images.
- **Modern Web Interface**: Built with React, Vite, and Tailwind CSS.
- **Fast Backend**: Powered by Flask with optimized AI model inference.

## 🛠️ Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, React Router.
- **Backend**: Python, Flask, PyTorch, Torchvision, Transformers.
- **Analysis**: C2PA (optional), Custom Ensemble AI Predictor.

## 📦 Installation & Setup

### Prerequisites

- Python 3.8+
- Node.js & npm

### 1. Clone the Repository

```bash
git clone https://github.com/christinawdc/AI-media-detection-tool.git
cd AI-media-detection-tool
```

### 2. Backend Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the backend:
   ```bash
   python backend/app.py
   ```
   The backend will start on `http://127.0.0.1:5000`.

4. Optional: enable video detection checkpoints:
   - Place `efficientnet.onnx` and `model.pth` in `backend/checkpoints/`.
   - Without these files, `/api/analyze_video` falls back to frame-based scoring with the existing image detector.

### 3. Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Build for production:
   ```bash
   npm run build
   ```
   Alternatively, run in development mode (proxies `/api` to `http://localhost:5000`):
   ```bash
   npm run dev
   ```

Note: the Flask app currently serves the UI from `backend/templates` and `backend/static`.

## 📂 Project Structure

- `backend/`: Flask application, AI models, and analysis logic.
- `frontend/`: React source code, components, and styling.
- `requirements.txt`: Python dependencies.
- `LICENSE`: Project license.

## 🛡️ License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
