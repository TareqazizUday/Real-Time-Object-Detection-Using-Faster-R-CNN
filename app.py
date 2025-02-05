import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import tempfile

# Check GPU or CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load Faster R-CNN Model
def create_model(num_classes, checkpoint=None, device='cpu'):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True,
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        pretrained_backbone=True,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)

# Object Detection Inference Class
class InferFasterRCNN:
    def __init__(self, num_classes, classnames):
        self.num_classes = num_classes
        self.classnames = ["__background__"] + classnames
        self.colors = np.random.uniform(0, 255, size=(len(self.classnames), 3))

    def load_model(self, checkpoint, device="cpu"):
        self.device = device
        self.model = create_model(self.num_classes, checkpoint=checkpoint, device=self.device)
        self.model.eval()

    def infer_frame(self, frame, detection_threshold=0.5):
        img_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        results = {"boxes": [], "scores": [], "classes": []}

        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].numpy()
            scores = outputs[0]["scores"].numpy()
            labels = outputs[0]["labels"].numpy()

            for i in range(len(boxes)):
                if scores[i] >= detection_threshold:
                    results["boxes"].append(boxes[i])
                    results["classes"].append(self.classnames[labels[i]])
                    results["scores"].append(scores[i])

        return results

# Class List (Modify based on your dataset)
class_list = [
    "Attention Please", "Beware of children", "Cycle route ahead warning",
    "End of all speed and passing limits", "Give Way", "Go Straight or Turn left",
    "Keep-Left", "Left Zig Zag Traffic", "Pedestrian", "Pedestrian Crossing",
    "Round-About", "Slippery road ahead", "Speed Limit 50 KMPh", "Stop_Sign",
    "Straight Ahead Only", "Traffic_signal", "Truck traffic is prohibited",
    "Turn left ahead", "Turn right ahead", "bike", "bus", "car", "cng", "truck"
]

# Load Model
IF_C = InferFasterRCNN(num_classes=len(class_list) + 1, classnames=class_list)
IF_C.load_model(checkpoint="D:/Codes/task/model/best_model.pth", device=device)

# STREAMLIT UI
st.title("Real-Time Object Detection")
st.write("Choose an option below to start detection.")

# Select Detection Mode
mode = st.radio("Select Detection Mode:", ["Multiple Images", "Video", "Webcam"])

# Process Multiple Images
if mode == "Multiple Images":
    uploaded_images = st.file_uploader("Upload Multiple Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_images:
        cols = st.columns(2)

        for index, uploaded_image in enumerate(uploaded_images):
            img = Image.open(uploaded_image).convert("RGB")
            img_np = np.array(img)

            results = IF_C.infer_frame(img_np)

            # Draw bounding boxes
            for i, box in enumerate(results["boxes"]):
                x1, y1, x2, y2 = map(int, box)
                label = f"{results['classes'][i]}"
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert image back to PIL format
            img_pil = Image.fromarray(img_np)

            # Show image in the corresponding column
            with cols[index % 2]:
                st.image(img_pil, caption=f"Detected Objects ({index+1})", use_container_width=True)
                st.write("### Detected Classes:")
                st.write(", ".join(results["classes"]))

# Process Uploaded Video
elif mode == "Video":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        # Save uploaded file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(uploaded_video.read())
        temp_file.close()

        st.write("Processing video...")

        # Open the video
        video = cv2.VideoCapture(temp_file.name)

        # Get video properties
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(video.get(cv2.CAP_PROP_FPS))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video
        output_video_path = temp_file.name.replace(".mp4", "_output.mp4")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        stframe = st.empty()
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = IF_C.infer_frame(frame_rgb)

            # Draw bounding boxes
            for i, box in enumerate(results["boxes"]):
                x1, y1, x2, y2 = map(int, box)
                label = f"{results['classes'][i]}"
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write frame to output video
            out.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            # Display processed frame
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        video.release()
        out.release()

        # Provide download button for processed video
        with open(output_video_path, "rb") as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")

# Process Webcam
elif mode == "Webcam":
    st.write("Opening webcam...")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to access webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = IF_C.infer_frame(frame_rgb)

        # Draw bounding boxes
        for i, box in enumerate(results["boxes"]):
            x1, y1, x2, y2 = map(int, box)
            label = f"{results['classes'][i]}"
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show live webcam feed
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
