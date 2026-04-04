import os
import sys
import time
import uuid
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# -------- PATH SETUP --------
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)
sys.path.insert(0, backend_dir)

# -------- FLASK APP (created immediately, no heavy work) --------
UPLOAD_FOLDER = os.path.join(backend_dir, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(
    __name__,
    template_folder=os.path.join(backend_dir, 'templates'),
    static_folder=os.path.join(backend_dir, 'static'),
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- GLOBAL MODEL STATE --------
predictor = None
synthid_detector = None
synthid_import_error = None
_models_ready = False
_models_loading = False


# -------- BACKGROUND INIT (download + load, runs after port binds) --------
def initialize_all():
    global predictor, synthid_detector, synthid_import_error
    global _models_ready, _models_loading

    _models_loading = True
    print("==> Background: Starting model download + load...")

    # Step 1: Download missing files from HuggingFace
    try:
        from huggingface_hub import hf_hub_download
        repo = "Adhil786/deepfake-models"
        token = os.environ.get("HF_TOKEN", None)

        files = [
            "model_output/synthid/robust_codebook.pkl",
            "model_output/resnet50_finetuned_benchmark.pth",
            "ai_detector_meta_learner.joblib",
            "polynomial_transformer.joblib",
        ]

        for filepath in files:
            full_path = os.path.join(project_root, filepath)
            if not Path(full_path).exists():
                print(f"==> Downloading {filepath}...")
                Path(full_path).parent.mkdir(parents=True, exist_ok=True)
                hf_hub_download(
                    repo_id=repo,
                    filename=filepath,
                    local_dir=project_root,
                    token=token,
                )
                print(f"==> ✅ Downloaded {filepath}")
            else:
                print(f"==> ✅ Already exists: {filepath}")
    except Exception as e:
        print(f"==> ❌ Download error: {e}")

    # Step 2: Load AI predictor
    try:
        from combine_model import AIEnsemblePredictor
        predictor = AIEnsemblePredictor()
        print("==> ✅ AI predictor loaded.")
    except Exception as e:
        print(f"==> ❌ AI predictor failed: {e}")

    # Step 3: Load SynthID
    try:
        from src.synthid import SynthIDService
        synthid_detector = SynthIDService(
            codebook_path=os.path.join(project_root, 'model_output', 'synthid', 'robust_codebook.pkl')
        )
        if synthid_detector.available:
            print("==> ✅ SynthID detector loaded.")
        else:
            print(f"==> ⚠️ SynthID unavailable: {synthid_detector.error}")
    except Exception as e:
        synthid_import_error = str(e)
        print(f"==> ❌ SynthID import failed: {e}")

    _models_loading = False
    _models_ready = True
    print("==> ✅ All models ready.")


# Start background init — port binds before this finishes
threading.Thread(target=initialize_all, daemon=True).start()


# -------- HELPERS --------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_temp_upload_path(original_filename: str) -> str:
    safe_name = secure_filename(original_filename) or "upload.bin"
    _, ext = os.path.splitext(safe_name)
    return os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4().hex}{ext.lower()}")


def analyze_synthid_layer(image_path):
    if synthid_detector is None:
        return {
            'status': 'unavailable',
            'available': False,
            'message': 'SynthID detector not yet loaded' if _models_loading else 'SynthID detector import failed',
            'error': synthid_import_error or 'SynthID not available',
        }
    return synthid_detector.analyze(image_path)


# -------- ROUTES --------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/video_dashboard')
def video_dashboard():
    return render_template('video_dashboard.html')


@app.route('/report')
def report():
    return render_template('report.html')


@app.route('/video_report')
def video_report():
    return render_template('video_report.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        from src.c2pa_checker import get_c2pa_runtime_status
        c2pa_status = get_c2pa_runtime_status()
    except Exception as e:
        c2pa_status = {'error': str(e)}

    return jsonify({
        'success': True,
        'status': 'ok',
        'python_executable': sys.executable,
        'python_version': sys.version,
        'runtime_initialized': _models_ready,
        'ai_init_mode': 'async',
        'models': {
            'ai_predictor_loaded': predictor is not None,
            'ai_predictor_initialized': predictor is not None,
            'ai_predictor_init_in_progress': _models_loading,
            'ai_predictor_error': None,
            'synthid_loaded': synthid_detector is not None and bool(getattr(synthid_detector, 'available', False)),
            'synthid_initialized': synthid_detector is not None,
            'synthid_init_in_progress': _models_loading,
            'synthid_error': synthid_import_error,
        },
        'c2pa': c2pa_status,
    })


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = _build_temp_upload_path(filename)
    file.save(filepath)

    result = {
        'success': True,
        'filename': filename,
        'layers': {'c2pa': None, 'synthid': None, 'ai_model': None},
        'explainability': None,
        'final_verdict': None,
        'confidence': 0.0,
        'is_ai_generated': False,
    }

    try:
        from src.c2pa_checker import check_c2pa

        time.sleep(1.5)
        c2pa_result = check_c2pa(filepath)
        result['layers']['c2pa'] = c2pa_result

        if c2pa_result.get('available') is False:
            result['layers']['c2pa']['status'] = 'unavailable'
        elif c2pa_result.get('c2pa_present'):
            result['layers']['c2pa']['status'] = 'verified' if c2pa_result.get('valid') else 'present'
        else:
            result['layers']['c2pa']['status'] = 'not_found'

        c2pa_metadata_present = bool(c2pa_result.get('c2pa_present'))

        if c2pa_metadata_present:
            result['confidence'] = 100.0
            result['is_ai_generated'] = True
            result['final_verdict'] = (
                'AI Generated (C2PA AI Declaration)'
                if c2pa_result.get('ai_generated')
                else 'AI Generated (C2PA Metadata Present)'
            )
            time.sleep(0.5)
            result['layers']['synthid'] = {'status': 'skipped', 'reason': 'C2PA metadata present'}
            time.sleep(0.5)
            result['layers']['ai_model'] = {'status': 'skipped', 'reason': 'C2PA metadata present'}
        else:
            time.sleep(1.0)
            synthid_result = analyze_synthid_layer(filepath)
            result['layers']['synthid'] = synthid_result
            synthid_detected = (
                synthid_result.get('status') == 'complete' and
                synthid_result.get('is_watermarked')
            )

            if synthid_detected:
                result['confidence'] = max(result['confidence'], synthid_result.get('confidence', 0.0))
                result['is_ai_generated'] = True
                result['final_verdict'] = 'AI Generated (SynthID Watermark Detected)'
                result['layers']['ai_model'] = {'status': 'skipped', 'reason': 'SynthID watermark detected'}
            else:
                time.sleep(2.0)
                if predictor is not None:
                    model_result = predictor.predict(filepath, return_details=True)
                    if model_result.get('status') == 'complete':
                        result['layers']['ai_model'] = {
                            'status': 'complete',
                            'label': model_result.get('label'),
                            'confidence': model_result.get('confidence_percent', 0.0),
                            'ai_probability': model_result.get('ai_probability_percent', 0.0),
                            'source': model_result.get('source'),
                            'model_scores': model_result.get('model_scores', {}),
                            'forensic_modules': model_result.get('forensic_modules', {}),
                        }
                        result['explainability'] = {'artifacts': model_result.get('artifacts', [])}
                    else:
                        result['layers']['ai_model'] = {
                            'status': 'error',
                            'error': model_result.get('error', 'AI model inference failed'),
                        }
                else:
                    result['layers']['ai_model'] = {
                        'status': 'error',
                        'error': 'AI model still loading, please try again shortly.' if _models_loading else 'AI model not loaded',
                    }

                ai_model_result = result['layers']['ai_model']
                if ai_model_result.get('status') == 'complete':
                    model_label = ai_model_result.get('label')
                    model_confidence = float(ai_model_result.get('confidence', 0.0))
                    model_scores = ai_model_result.get('model_scores', {}) or {}
                    forensic_modules = ai_model_result.get('forensic_modules', {}) or {}
                    supporting_modules = forensic_modules.get('supporting_modules', []) or []
                    forensics_only_score = float(model_scores.get('forensics_only_ai_score', 0.0))

                    low_confidence_override = (
                        model_label == 'AI Image' and
                        model_confidence < 60.0 and
                        not supporting_modules and
                        forensics_only_score < 45.0 and
                        synthid_result.get('status') == 'complete' and
                        not synthid_result.get('is_watermarked')
                    )

                    if low_confidence_override:
                        ai_model_result['pipeline_override'] = {
                            'applied': True,
                            'reason': 'Low-confidence model-only AI prediction with negative SynthID and weak forensic support',
                        }
                        result['confidence'] = max(50.0, min(75.0, 100.0 - forensics_only_score))
                        result['is_ai_generated'] = False
                        result['final_verdict'] = 'Real Image'
                    else:
                        ai_model_result['pipeline_override'] = {'applied': False}
                        result['confidence'] = model_confidence
                        result['is_ai_generated'] = model_label == 'AI Image'
                        result['final_verdict'] = model_label or 'Unknown'
                elif c2pa_result.get('c2pa_present'):
                    result['final_verdict'] = 'Unknown (Signed provenance present)'
                else:
                    result['final_verdict'] = 'Unknown (Model unavailable)'

    except Exception as exc:
        result['success'] = False
        result['error'] = str(exc)
    else:
        try:
            from forensic import build_image_forensic_summary
            result['forensic_summary'] = build_image_forensic_summary(result)
        except Exception:
            pass
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    allowed_video_ext = {'mp4', 'avi', 'mov'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_video_ext:
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    filename = secure_filename(file.filename)
    filepath = _build_temp_upload_path(filename)
    file.save(filepath)

    try:
        from video_detect_standalone import deepfakes_video_predict
        from forensic import build_video_forensic_summary

        video_result = deepfakes_video_predict(filepath, predictor=predictor)
        if isinstance(video_result, dict):
            result = {'success': True, **video_result}
            result['forensic_summary'] = build_video_forensic_summary(result)
        else:
            result = {'success': True, 'result': str(video_result)}
    except Exception as exc:
        result = {'success': False, 'error': str(exc)}
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


@app.route('/api/forensic-report', methods=['POST'])
def get_forensic_report():
    try:
        analysis_result = request.get_json()
        if not analysis_result:
            return jsonify({'success': False, 'error': 'No analysis data provided'}), 400

        from forensic import generate_forensic_report
        report = generate_forensic_report(analysis_result)
        return jsonify(report)
    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# -------- RUN (dev only, Render uses serve.py) --------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*50}\nDeepFake Defender Backend Running\n{'='*50}")
    print(f"Open http://0.0.0.0:{port} in your browser\n")
    app.run(host="0.0.0.0", debug=False, port=port)