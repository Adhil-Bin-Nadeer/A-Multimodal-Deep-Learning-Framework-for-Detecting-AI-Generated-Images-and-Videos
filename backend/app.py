import os
import sys
import time
import threading
import uuid

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename


# Add backend to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(backend_dir)

sys.path.insert(0, backend_dir)

from src.c2pa_checker import check_c2pa, get_c2pa_runtime_status
from forensic import (
    build_image_forensic_summary,
    build_video_forensic_summary,
    generate_forensic_report,
)


# -------- CONFIG --------
UPLOAD_FOLDER = os.path.join(backend_dir, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(
    __name__,
    template_folder=os.path.join(backend_dir, 'templates'),
    static_folder=os.path.join(backend_dir, 'static'),
)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SIMULATE_LAYER_DELAYS = os.environ.get("SIMULATE_LAYER_DELAYS", "0") == "1"


def _maybe_delay(seconds: float) -> None:
    if SIMULATE_LAYER_DELAYS and seconds > 0:
        time.sleep(seconds)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _build_temp_upload_path(original_filename: str) -> str:
    safe_name = secure_filename(original_filename) or "upload.bin"
    _, ext = os.path.splitext(safe_name)
    temp_name = f"{uuid.uuid4().hex}{ext.lower()}"
    return os.path.join(app.config['UPLOAD_FOLDER'], temp_name)


def analyze_synthid_layer(image_path):
    _initialize_synthid_detector()

    if synthid_detector is None:
        return {
            'status': 'unavailable',
            'available': False,
            'message': 'SynthID detector unavailable',
            'error': synthid_import_error or 'SynthID detector is not available in this runtime',
        }

    return synthid_detector.analyze(image_path)


predictor = None
synthid_detector = None
synthid_import_error = None
predictor_init_error = None
predictor_initialized = False
synthid_initialized = False
predictor_init_in_progress = False
synthid_init_in_progress = False
predictor_init_lock = threading.Lock()
synthid_init_lock = threading.Lock()


def _initialize_ai_predictor():
    global predictor
    global predictor_init_error
    global predictor_initialized
    global predictor_init_in_progress

    if predictor_initialized:
        return

    with predictor_init_lock:
        if predictor_initialized:
            return

        predictor_init_in_progress = True
        try:
            print("Initializing AI Detection Models...")
            from combine_model import AIEnsemblePredictor

            predictor = AIEnsemblePredictor()
            predictor_init_error = None
            print("Models loaded successfully.")
        except Exception as exc:
            predictor = None
            predictor_init_error = str(exc)
            print(f"Warning: Could not load AI models: {exc}")
            print("C2PA checking will still work, but AI detection will be unavailable.")
        finally:
            predictor_initialized = True
            predictor_init_in_progress = False


def _initialize_synthid_detector():
    global synthid_detector
    global synthid_import_error
    global synthid_initialized
    global synthid_init_in_progress

    if synthid_initialized:
        return

    with synthid_init_lock:
        if synthid_initialized:
            return

        synthid_init_in_progress = True
        try:
            print("Initializing SynthID Detector...")
            try:
                from src.synthid import SynthIDService
                synthid_import_error = None
            except Exception as exc:
                SynthIDService = None
                synthid_import_error = str(exc)

            if SynthIDService is None:
                synthid_detector = None
                print(f"Warning: SynthID detector import failed: {synthid_import_error}")
            else:
                synthid_detector = SynthIDService(
                    codebook_path=os.path.join(project_root, 'model_output', 'synthid', 'robust_codebook.pkl')
                )

                if synthid_detector.available:
                    print("SynthID detector loaded successfully.")
                else:
                    print(f"Warning: SynthID detector unavailable: {synthid_detector.error}")
        finally:
            synthid_initialized = True
            synthid_init_in_progress = False



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
    c2pa_status = get_c2pa_runtime_status()
    runtime_initialized = predictor_initialized or synthid_initialized

    return jsonify(
        {
            'success': True,
            'status': 'ok',
            'runtime_initialized': runtime_initialized,
            'python_executable': sys.executable,
            'python_version': sys.version,
            'models': {
                'ai_predictor_loaded': predictor is not None,
                'ai_predictor_error': predictor_init_error,
                'ai_predictor_initialized': predictor_initialized,
                'ai_predictor_init_in_progress': predictor_init_in_progress,
                'synthid_loaded': synthid_detector is not None and bool(getattr(synthid_detector, 'available', False)),
                'synthid_error': synthid_import_error if synthid_detector is None else getattr(synthid_detector, 'error', None),
                'synthid_initialized': synthid_initialized,
                'synthid_init_in_progress': synthid_init_in_progress,
            },
            'c2pa': c2pa_status,
        }
    )


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Main analysis endpoint.
    Pipeline: C2PA Check -> SynthID -> AI Model
    """
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
        'layers': {
            'c2pa': None,
            'synthid': None,
            'ai_model': None,
        },
        'explainability': None,
        'final_verdict': None,
        'confidence': 0.0,
        'is_ai_generated': False,
    }

    try:
        # ========== LAYER 1: C2PA CHECK ==========
        _maybe_delay(1.5)
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
            if c2pa_result.get('ai_generated'):
                result['final_verdict'] = 'AI Generated (C2PA AI Declaration)'
            else:
                result['final_verdict'] = 'AI Generated (C2PA Metadata Present)'

            _maybe_delay(0.5)
            result['layers']['synthid'] = {'status': 'skipped', 'reason': 'C2PA metadata present'}
            _maybe_delay(0.5)
            result['layers']['ai_model'] = {'status': 'skipped', 'reason': 'C2PA metadata present'}
        else:
            # ========== LAYER 2: SYNTHID ==========
            _maybe_delay(1.0)
            synthid_result = analyze_synthid_layer(filepath)
            result['layers']['synthid'] = synthid_result
            synthid_detected = (
                synthid_result.get('status') == 'complete' and
                synthid_result.get('is_watermarked')
            )

            if synthid_detected:
                result['confidence'] = max(
                    result['confidence'],
                    synthid_result.get('confidence', 0.0),
                )
                result['is_ai_generated'] = True
                result['final_verdict'] = 'AI Generated (SynthID Watermark Detected)'
                result['layers']['ai_model'] = {
                    'status': 'skipped',
                    'reason': 'SynthID watermark detected',
                }
            else:
                # ========== LAYER 3: AI MODEL ==========
                _maybe_delay(2.0)

                if predictor is None:
                    _initialize_ai_predictor()

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
                        result['explainability'] = {
                            'artifacts': model_result.get('artifacts', []),
                        }
                    else:
                        result['layers']['ai_model'] = {
                            'status': 'error',
                            'error': model_result.get('error', 'AI model inference failed'),
                        }
                else:
                    result['layers']['ai_model'] = {
                        'status': 'error',
                        'error': predictor_init_error or 'AI model not loaded',
                    }

                ai_model_result = result['layers']['ai_model']
                if ai_model_result.get('status') == 'complete':
                    model_label = ai_model_result.get('label')
                    model_confidence = float(ai_model_result.get('confidence', 0.0))
                    model_scores = ai_model_result.get('model_scores', {}) or {}
                    forensic_modules = ai_model_result.get('forensic_modules', {}) or {}
                    supporting_modules = forensic_modules.get('supporting_modules', []) or []
                    forensics_only_score = float(model_scores.get('forensics_only_ai_score', 0.0))

                    low_confidence_model_only_ai = (
                        model_label == 'AI Image' and
                        model_confidence < 60.0 and
                        not supporting_modules and
                        forensics_only_score < 45.0 and
                        synthid_result.get('status') == 'complete' and
                        not synthid_result.get('is_watermarked')
                    )

                    if low_confidence_model_only_ai:
                        ai_model_result['pipeline_override'] = {
                            'applied': True,
                            'reason': 'Low-confidence model-only AI prediction with negative SynthID and weak forensic support',
                        }
                        result['confidence'] = max(50.0, min(75.0, 100.0 - forensics_only_score))
                        result['is_ai_generated'] = False
                        result['final_verdict'] = 'Real Image'
                    else:
                        ai_model_result['pipeline_override'] = {
                            'applied': False,
                        }
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
        result['forensic_summary'] = build_image_forensic_summary(result)

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """
    Video deepfake detection endpoint.
    Accepts a video file, runs detection, and returns the result.
    """
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

    dedicated_onnx = os.path.join(backend_dir, 'checkpoints', 'efficientnet.onnx')
    dedicated_pth = os.path.join(backend_dir, 'checkpoints', 'model.pth')
    has_dedicated_checkpoints = os.path.exists(dedicated_onnx) and os.path.exists(dedicated_pth)

    if predictor is None and not has_dedicated_checkpoints:
        _initialize_ai_predictor()

    try:
        from video_detect_standalone import deepfakes_video_predict

        video_result = deepfakes_video_predict(
            filepath,
            predictor=predictor,
            allow_predictor_autoload=False,
        )
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
    """
    Generate an enhanced forensic report using Gemini AI.
    Expects the analysis result JSON in the request body.
    """
    try:
        analysis_result = request.get_json()
        if not analysis_result:
            return jsonify({'success': False, 'error': 'No analysis data provided'}), 400

        report = generate_forensic_report(analysis_result)
        return jsonify(report)

    except Exception as exc:
        return jsonify({'success': False, 'error': str(exc)}), 500


# -------- RUN --------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    print("\n" + "=" * 50)
    print("DeepFake Defender Backend Running")
    print("=" * 50)
    print(f"Open http://0.0.0.0:{port} in your browser\n")
    app.run(host="0.0.0.0", debug=False, port=port)
