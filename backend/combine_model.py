import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np
import joblib

# -------- CONFIG --------
# Paths relative to the project root (one level up from backend/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESNET_PATH = os.path.join(_BASE_DIR, "model_output", "resnet50_finetuned_benchmark.pth")
META_LEARNER_PATH = os.path.join(_BASE_DIR, "ai_detector_meta_learner.joblib")
POLY_TRANSFORM_PATH = os.path.join(_BASE_DIR, "polynomial_transformer.joblib")

VIT_NAME = "dima806/ai_vs_real_image_detection"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AIEnsemblePredictor:
    def __init__(self):
        print(f"⏳ Loading models to {DEVICE}...")
        self.device = DEVICE
        
        # 1. Load ResNet50
        self.resnet = models.resnet50(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 2)
        
        if os.path.exists(RESNET_PATH):
            self.resnet.load_state_dict(torch.load(RESNET_PATH, map_location=DEVICE))
            print("✅ ResNet loaded.")
        else:
            raise FileNotFoundError(f"Could not find ResNet model at {RESNET_PATH}")
            
        self.resnet.to(self.device).eval()
        
        # ResNet Preprocessing
        self.res_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Load ViT
        self.vit = AutoModelForImageClassification.from_pretrained(VIT_NAME).to(self.device).eval()
        self.vit_processor = AutoImageProcessor.from_pretrained(VIT_NAME)
        print("✅ ViT loaded.")

        # 3. Load Meta-Learner (Polynomial + Logistic Regression)
        if os.path.exists(META_LEARNER_PATH) and os.path.exists(POLY_TRANSFORM_PATH):
            self.meta_model = joblib.load(META_LEARNER_PATH)
            self.poly = joblib.load(POLY_TRANSFORM_PATH)
            print("✅ Meta-Learner loaded.")
        else:
            raise FileNotFoundError("Meta-learner files not found! Run the training script first to generate .joblib files.")

    def predict(self, image_path):
        if not os.path.exists(image_path):
            return "Error", f"Image not found at {image_path}"

        # Open Image
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return "Error", f"Invalid image file: {e}"

        with torch.no_grad():
            # --- ResNet Prediction ---
            res_input = self.res_transform(img).unsqueeze(0).to(self.device)
            res_logits = self.resnet(res_input)
            res_prob = torch.softmax(res_logits, dim=1)[0, 1].item()

            # --- ViT Prediction ---
            vit_inputs = self.vit_processor(images=img, return_tensors="pt").to(self.device)
            vit_logits = self.vit(**vit_inputs).logits
            vit_prob = torch.softmax(vit_logits, dim=1)[0, 1].item()

        # --- Meta-Learner Ensemble ---
        raw_scores = np.array([[res_prob, vit_prob]])
        poly_features = self.poly.transform(raw_scores)
        final_ai_prob = self.meta_model.predict_proba(poly_features)[0, 1]
        
        if final_ai_prob > 0.5:
            label = "AI Image"
            confidence = final_ai_prob
        else:
            label = "Real Image"
            confidence = 1 - final_ai_prob
        
        return label, confidence


# -------- EXECUTION --------
if __name__ == "__main__":
    predictor = AIEnsemblePredictor()

    print("\n" + "="*40)
    print("🤖 SINGLE IMAGE DETECTOR READY")
    print("="*40)

    while True:
        img_path = input("\nEnter path to image (or 'q' to quit): ").strip().strip('"')
        
        if img_path.lower() == 'q':
            break
            
        label, score = predictor.predict(img_path)
        
        if label == "Error":
            print(f"❌ {score}")
        else:
            color = "\033[91m" if label == "AI Image" else "\033[92m"
            reset = "\033[0m"
            print(f"Prediction: {color}{label}{reset}")
            print(f"Confidence: {score*100:.2f}%")
            print("-" * 20)
