import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from PIL import Image

# Suppress HuggingFace warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
import warnings
warnings.filterwarnings("ignore")

# --- 1. Setup & Config ---
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = 'fake_avatar.mp4' if os.path.exists('fake_avatar.mp4') else 'fake_video.mp4'
NUM_FRAMES_TO_PROCESS = 100 # ML inference is heavier, so we cap at 100 frames
PROCESS_EVERY_N_FRAMES = 3 # Process every 3rd frame to save time
FPS = 30 

print("Loading Deepfake Detection ML Model... (This may take a moment)")
try:
    from transformers import pipeline
    # Load the image classification pipeline with a different, highly-rated model
    detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
except ImportError:
    print("Error: 'transformers' or 'torch' is not installed.")
    print("Please run: pip install torch torchvision transformers pillow")
    sys.exit(1)

# Initialize OpenCV Face Detection (built-in, no external dependencies)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- 2. Video Loop & Data Collection ---
frame_count = 0
processed_count = 0
fake_scores = []
frame_indices = []

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video: {VIDEO_PATH}")
    sys.exit(1)

print(f"Analyzing video: {VIDEO_PATH}")

while cap.isOpened() and processed_count < NUM_FRAMES_TO_PROCESS:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    frame_height, frame_width, _ = frame.shape
    
    if len(faces) > 0:
        # Use the first face found
        (x, y, w, h) = faces[0]
        
        # Add some padding around the face
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_width, x + w + pad_x)
        y2 = min(frame_height, y + h + pad_y)
        
        # Crop the face
        face_roi = frame_rgb[y1:y2, x1:x2]
        
        if face_roi.size > 0:
            # Convert to PIL Image for the HuggingFace pipeline
            pil_image = Image.fromarray(face_roi)
            
            # Run inference
            try:
                predictions = detector(pil_image)
                
                # predictions is a list of dicts: [{'label': 'FAKE', 'score': 0.9}, {'label': 'REAL', 'score': 0.1}]
                # Or sometimes just [{'label': 'real', 'score': 0.9}, ...]
                fake_score = 0.0
                for pred in predictions:
                    label = pred['label'].upper()
                    if 'FAKE' in label:
                        fake_score = pred['score']
                    elif 'REAL' in label and fake_score == 0.0:
                        # If it only outputs REAL score and no FAKE score found yet
                        fake_score = 1.0 - pred['score']
                        
                fake_scores.append(fake_score)
                frame_indices.append(frame_count)
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{NUM_FRAMES_TO_PROCESS} frames...")
            except Exception as e:
                print(f"Warning: Inference failed for frame {frame_count} - {e}")
                pass

cap.release()

print(f"Extraction complete. Evaluated {len(fake_scores)} frames.")

# --- 3. Analysis & Verdict ---
if len(fake_scores) == 0:
    print("No faces detected in the video. Cannot determine authenticity.")
    sys.exit(0)

# Video-level score: use the 90th percentile instead of flat average
# Deepfakes often have a few highly-glitched frames while the rest look real.
# The 90th percentile captures this high-confidence signal much better than an average.
avg_fake_score = np.percentile(fake_scores, 90)
max_fake_score = np.max(fake_scores)

# --- 4. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(frame_indices, fake_scores, marker='o', linestyle='-', color='b', label='AI-Generated Confidence')
plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (50%)')
plt.title(f'Deepfake Detection Confidence Over Time\nAverage AI Confidence: {avg_fake_score*100:.1f}%')
plt.xlabel('Frame Number')
plt.ylabel('AI Confidence Score (0.0 to 1.0)')
plt.ylim(-0.05, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()

output_img = os.path.splitext(VIDEO_PATH)[0] + "_ml_analysis.png"
plt.savefig(output_img)
print(f"Saved analysis plot to: {output_img}")
plt.close()

# --- 5. Verdict ---
verdict = "REAL"
reason = []

if avg_fake_score > 0.5:
    verdict = "FAKE"
    reason.append(f"Average AI-generated confidence ({avg_fake_score*100:.1f}%) is greater than 50%")
elif max_fake_score > 0.8:
    verdict = "SUSPICIOUS"
    reason.append(f"While average is low, some frames had very high AI confidence (Peak: {max_fake_score*100:.1f}%)")

print("\n--- DEEPFAKE DETECTION VERDICT ---")
print(f"VERDICT: {verdict}")
if reason:
    print("Reason(s):")
    for r in reason:
        print(f"- {r}")
else:
    print(f"High confidence the video is original (AI Score: {avg_fake_score*100:.1f}%).")
print("----------------------------------")