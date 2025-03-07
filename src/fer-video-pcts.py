import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import time
import pandas as pd  # To save logs

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the emotion detection model
model_path = r'C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\models\model_vicfiltered_notransforms_lr5e4_1000_6_emotions_res18_ai.pth'

# Define model architecture (MUST match training)
model = timm.create_model('resnet18', pretrained=False, num_classes=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define emotion labels
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define preprocessing (Match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((252, 252)),  # Match training input size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Match training normalization
])

# Open webcam
cap = cv2.VideoCapture(0)
start_time = time.time()  # Start recording time

emotion_log = []  # Store emotion timestamps

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = round(time.time() - start_time, 2)  # Timestamp in seconds

    # Convert frame to RGB (OpenCV loads as BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = rgb_frame[y:y+h, x:x+w]  # Crop face (keep RGB)
        face_pil = Image.fromarray(face)  # Convert to PIL Image
        face_tensor = transform(face_pil).unsqueeze(0).to(device)  # Add batch dimension & move to device

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            
            # Get top emotion and its probability
            top_prob, top_index = torch.max(probabilities, 1)
            top_prob = top_prob.item() * 100  # Convert to percentage
            top_emotion = EMOTIONS[top_index.item()]

            # Log timestamp and emotion
            emotion_log.append({"Time (s)": current_time, "Emotion": top_emotion, "Confidence (%)": round(top_prob, 2)})

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{top_emotion}: {top_prob:.1f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Emotion Detector", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save the emotion log to a CSV file
log_df = pd.DataFrame(emotion_log)
log_file = r"C:\Users\Victor Cardenas\Documents\msc\semestre-4\idi-4\fer-2025\results\logs\emotion_log.csv"
log_df.to_csv(log_file, index=False)

print(f"Emotion log saved to {log_file}")
