import streamlit as st
import cv2
import onnxruntime as ort
import tempfile
import os
import numpy as np

# Load YOLO ONNX model
def load_model(onnx_path):
    session = ort.InferenceSession(onnx_path)
    return session

# Preprocess frame for YOLO model
def preprocess_frame(frame, input_shape):
    resized = cv2.resize(frame, input_shape)
    blob = cv2.dnn.blobFromImage(resized, scalefactor=1/255.0, size=input_shape, swapRB=True)
    return blob

# Post-process YOLO model outputs
def postprocess_outputs(outputs, input_shape, threshold=0.5):
    classes = []
    confidences = []
    boxes = []
    for detection in outputs:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > threshold:
            # Scale box to original image size
            center_x, center_y, width, height = detection[0:4]
            x = int((center_x - width / 2) * input_shape[0])
            y = int((center_y - height / 2) * input_shape[1])
            w = int(width * input_shape[0])
            h = int(height * input_shape[1])
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            classes.append(class_id)
    return boxes, confidences, classes

# Process video and detect objects
def process_video(video_path, model_session, input_shape):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_segments = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blob = preprocess_frame(frame, input_shape)
        outputs = model_session.run(None, {model_session.get_inputs()[0].name: blob})
        boxes, confidences, classes = postprocess_outputs(outputs[0], input_shape)
        if len(classes) > 0:
            frame_segments.append({"frame": cap.get(cv2.CAP_PROP_POS_FRAMES), "classes": classes})
    cap.release()
    return frame_segments

# Streamlit UI
st.title("YOLO Object Detection on Video")

# File uploader
uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])
onnx_path = st.text_input("Path to YOLO ONNX Model", value="yolov4.onnx")

if uploaded_file and onnx_path:
    st.write("Processing video...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Load model
    yolo_model = load_model(onnx_path)
    
    # Process video
    input_shape = (640, 640)  # Adjust based on your YOLO model
    detections = process_video(video_path, yolo_model, input_shape)

    # Display results
    st.write(f"Total segments with objects: {len(detections)}")
    for detection in detections:
        frame_no = detection["frame"]
        classes_detected = detection["classes"]
        st.write(f"Frame: {frame_no}, Classes: {classes_detected}")

    os.remove(video_path)
