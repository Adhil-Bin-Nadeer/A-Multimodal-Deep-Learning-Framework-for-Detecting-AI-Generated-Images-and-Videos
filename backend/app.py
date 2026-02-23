import os
import sys
import time
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# -------- PATH SETUP --------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)

# Add backend/src to path for internal imports
sys.path.insert(0, os.path.join(_BACKEND_DIR, 'src'))

from c2pa_checker import check_c2pa
from combine_model import AIEnsemblePredictor
from forensic import generate_forensic_report

# -------- CONFIG --------
UPLOAD_FOLDER = os.path.join(_BACKEND_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# React build output lives at <project_root>/frontend/dist
REACT_BUILD = os.path.join(_PROJECT_ROOT, 'frontend', 'dist')

app = Flask(__name__, static_folder=REACT_BUILD, static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load AI model once at startup
print("🚀 Initializing AI Detection Models...")
predictor = None
try:
    predictor = AIEnsemblePredictor()
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load AI models: {e}")
    print("   C2PA checking will still work, but AI detection will be unavailable.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------- ROUTES --------
# Serve React SPA for all non-API routes
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    # Serve existing static asset if present (JS, CSS, icons, etc.)
    full = os.path.join(REACT_BUILD, path)
    if path and os.path.exists(full):
        return send_from_directory(REACT_BUILD, path)
    # Fall back to index.html so React Router handles the route
    return send_from_directory(REACT_BUILD, 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """
    Main analysis endpoint.
    Pipeline: C2PA Check → (SynthID - skipped) → AI Model
    """
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'File type not allowed'}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = {
        'success': True,
        'filename': filename,
        'layers': {
            'c2pa': None,
            'synthid': None,  # Skipped for now
            'ai_model': None
        },
        'final_verdict': None,
        'confidence': 0,
        'is_ai_generated': False
    }

    try:
        # ========== LAYER 1: C2PA CHECK ==========
        time.sleep(1.5)
        c2pa_result = check_c2pa(filepath)
        result['layers']['c2pa'] = c2pa_result

        if c2pa_result.get('available') == False:
            result['layers']['c2pa']['status'] = 'unavailable'
        
        if c2pa_result.get('c2pa_present'):
            result['confidence'] = 100.0
            result['is_ai_generated'] = True
            result['final_verdict'] = 'AI Generated (C2PA Verified)'
            result['layers']['c2pa']['status'] = 'verified'
            
            time.sleep(0.5)
            result['layers']['synthid'] = {'status': 'skipped', 'reason': 'C2PA verification successful'}
            time.sleep(0.5)
            result['layers']['ai_model'] = {'status': 'skipped', 'reason': 'C2PA verification successful'}
            
        else:
            # ========== LAYER 2: SYNTHID (SKIPPED) ==========
            time.sleep(1.0)
            result['layers']['synthid'] = {'status': 'skipped', 'reason': 'Not implemented'}
            
            # ========== LAYER 3: AI MODEL ==========
            time.sleep(2.0)
            if predictor is not None:
                label, confidence = predictor.predict(filepath)
                confidence_percent = confidence * 100
                
                result['layers']['ai_model'] = {
                    'status': 'complete',
                    'label': label,
                    'confidence': confidence_percent
                }
                
                result['confidence'] = confidence_percent
                result['is_ai_generated'] = label == 'AI Image'
                result['final_verdict'] = label
            else:
                result['layers']['ai_model'] = {
                    'status': 'error',
                    'error': 'AI model not loaded'
                }
                result['final_verdict'] = 'Unknown (Model unavailable)'

    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
    
    finally:
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify(result)


@app.route('/api/forensic-report', methods=['POST'])
def get_forensic_report():
    """
    Generate an enhanced forensic report using the forensic module.
    Expects the analysis result JSON in the request body.
    """
    try:
        analysis_result = request.get_json()
        if not analysis_result:
            return jsonify({'success': False, 'error': 'No analysis data provided'}), 400
        
        report = generate_forensic_report(analysis_result)
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# -------- RUN --------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🛡️  DeepFake Defender Backend Running")
    print("="*50)
    print("Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
