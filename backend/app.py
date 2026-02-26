import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Add src to path for c2pa_checker import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from c2pa_checker import check_c2pa
from combine_model import AIEnsemblePredictor
from forensic import generate_forensic_report

# -------- CONFIG --------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

app = Flask(__name__, template_folder='templates', static_folder='static')
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
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/report')
def report():
    return render_template('report.html')


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
        time.sleep(1.5)  # Simulated processing time
        c2pa_result = check_c2pa(filepath)
        result['layers']['c2pa'] = c2pa_result

        # Check if C2PA library is available on this platform
        if c2pa_result.get('available') == False:
            result['layers']['c2pa']['status'] = 'unavailable'
        
        if c2pa_result.get('c2pa_present'):
            # C2PA metadata found - this means AI generated (AI tools add C2PA marks)
            result['confidence'] = 100.0
            result['is_ai_generated'] = True
            result['final_verdict'] = 'AI Generated (C2PA Verified)'
            result['layers']['c2pa']['status'] = 'verified'
            
            # Skip other layers since we have cryptographic proof
            time.sleep(0.5)
            result['layers']['synthid'] = {'status': 'skipped', 'reason': 'C2PA verification successful'}
            time.sleep(0.5)
            result['layers']['ai_model'] = {'status': 'skipped', 'reason': 'C2PA verification successful'}
            
        else:
            # ========== LAYER 2: SYNTHID (SKIPPED) ==========
            time.sleep(1.0)  # Simulated processing time
            result['layers']['synthid'] = {'status': 'skipped', 'reason': 'Not implemented'}
            
            # ========== LAYER 3: AI MODEL ==========
            time.sleep(2.0)  # Simulated model loading/inference time
            print(f"[DEBUG] predictor is None: {predictor is None}")
            if predictor is not None:
                print(f"[DEBUG] Calling predictor.predict({filepath})")
                label, confidence = predictor.predict(filepath)
                print(f"[DEBUG] Result: label={label}, confidence={confidence}")
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
    Generate an enhanced forensic report using Gemini AI.
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
    import os
    port = int(os.environ.get("PORT", 7860))
    print("\n" + "="*50)
    print("🛡️  DeepFake Defender Backend Running")
    print("="*50)
    print(f"Open http://0.0.0.0:{port} in your browser\n")
    app.run(host="0.0.0.0", debug=False, port=port)
