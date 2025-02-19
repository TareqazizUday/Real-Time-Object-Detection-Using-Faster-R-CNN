## Real-Time Object Detection Using Faster R-CNN

### Overview
This project implements a **real-time object detection system** using **Faster R-CNN with ResNet-50**. It allows users to detect objects in images, videos, and live webcam feeds using a **trained model**.

### Dataset Link: 
```
https://app.roboflow.com/containing-various-objects-detection-task/various-objects-detection-task/2
```
### Features
- Detect objects in **multiple images**.
- Perform real-time detection on **videos**.
- Use **live webcam feed** for object detection.
- Supports a **custom-trained Faster R-CNN model**.
- **Streamlit-based UI** for easy interaction.

### Requirements
- Python 3.8+
- PyTorch
- OpenCV (`cv2`)
- NumPy
- PIL (Pillow)
- Streamlit
- torchvision

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/TareqazizUday/Real-Time-Object-Detection-Using-Faster-R-CNN.git
   cd your-repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### How to Run
1. Place the trained model checkpoint (`best_model.pth`) inside the project directory.
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Usage
- **Multiple Images**: Upload multiple images for object detection.
- **Video**: Upload a video to process and detect objects.
- **Webcam**: Open the live webcam feed and detect objects in real time.

### Model Details
- Uses **Faster R-CNN ResNet-50 FPN** as the object detection model.
- Supports **custom object classes**.
- Loads a **pretrained checkpoint** for inference.

### Results
Multiple Image Results

![Picture2](https://github.com/user-attachments/assets/7b6b10fb-0ac2-412a-ac35-ece0ba3d4f15)

Webcam Results

![Picture1](https://github.com/user-attachments/assets/26982a3f-95a6-4b8d-86fa-1d25e1344d54)
![app-02-06-2025_12_08_AM](https://github.com/user-attachments/assets/5ed76f6a-37ae-4062-b0db-bd8f74979f49)

### Notes
- Ensure the model checkpoint is correctly placed in the project directory.
- GPU is recommended for faster inference.
- Modify the `class_list` in the script to match your dataset.

### License
This project is open-source. Feel free to modify and use it as needed.

---
