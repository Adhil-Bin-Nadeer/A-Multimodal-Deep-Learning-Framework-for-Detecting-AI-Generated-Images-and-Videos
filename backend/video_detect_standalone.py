import cv2
import torch
import numpy as np
from models import image
from onnx2pytorch import ConvertModel
import onnx

# Predefined video path (change this to your video file)
VIDEO_PATH = 'uploads/videoplayback.mp4'  # Update this path to your actual video file

# Load ONNX model and convert to PyTorch
onnx_model = onnx.load('backend/checkpoints/efficientnet.onnx')
pytorch_model = ConvertModel(onnx_model)

# Load PyTorch weights for the encoder
ckpt = torch.load('backend/checkpoints/model.pth', map_location=torch.device('cpu'))
pytorch_model.load_state_dict(ckpt['rgb_encoder'], strict=True)
pytorch_model.eval()

def preprocess_img(frame):
    frame = frame / 255.0
    frame = cv2.resize(frame, (256, 256))
    face_pt = torch.unsqueeze(torch.Tensor(frame), dim=0)
    return face_pt

def preprocess_video(input_video, n_frames=3):
    v_cap = cv2.VideoCapture(input_video)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_img(frame)
            frames.append(frame)
    v_cap.release()
    return frames

def deepfakes_video_predict(input_video):
    video_frames = preprocess_video(input_video)
    real_faces_list = []
    fake_faces_list = []
    for face in video_frames:
        img_grads = pytorch_model.forward(face)
        img_grads = img_grads.cpu().detach().numpy()
        img_grads_np = np.squeeze(img_grads)
        real_faces_list.append(img_grads_np[0])
        fake_faces_list.append(img_grads_np[1])
    real_faces_mean = np.mean(real_faces_list)
    fake_faces_mean = np.mean(fake_faces_list)
    if real_faces_mean > 0.5:
        preds = round(real_faces_mean * 100, 3)
        text2 = f"The video is REAL. Confidence score: {preds}%"
    else:
        preds = round(fake_faces_mean * 100, 3)
        text2 = f"The video is FAKE. Confidence score: {preds}%"
    return text2

if __name__ == '__main__':
    import os
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
        print("Please place your video file in the uploads folder or update VIDEO_PATH")
    else:
        result = deepfakes_video_predict(VIDEO_PATH)
        print(result)
