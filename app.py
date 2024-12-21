import streamlit as st
import cv2
import onnxruntime as ort
import tempfile
import os
import numpy as np
from ultralytics import YOLO
import time


def process_video(video_path, model, selected_ind):
    videocapture = cv2.VideoCapture(video_path)  # Capture the video
    if not videocapture.isOpened():
        st.error("Could not open webcam.")

    fps_display = st.sidebar.empty() 
    stop_button = st.button("Stop")  
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            st.warning("Failed to read frame from webcam. Please make sure the webcam is connected properly.")
            break

        prev_time = time.time()  # Store initial time for FPS calculation

        # Store model predictions
        # if enable_trk == "Yes":
        #     results = model.track(frame, conf=conf, iou=iou, classes=selected_ind, persist=True)
        # else:
        results = model(frame, conf=conf, iou=iou, classes=selected_ind)
        annotated_frame = results[0].plot()  # Add annotations on frame


        # Calculate model FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)

        # display frame
        org_frame.image(frame, channels="BGR")
        ann_frame.image(annotated_frame, channels="BGR")

        if stop_button:
            videocapture.release()  # Release the capture
            torch.cuda.empty_cache()  # Clear CUDA memory
            st.stop()  # Stop streamlit app

        # Display FPS in sidebar
        fps_display.metric("FPS", f"{fps:.2f}")

    # Release the capture
    videocapture.release()

    # Clear CUDA memory
    torch.cuda.empty_cache()

    # Destroy window
    cv2.destroyAllWindows()

def process_image(image_path, model, selected_ind):

    results = model([image_path], conf=conf, iou=iou, classes=selected_ind)
    # Process results list
    for result in results:
        org_frame.image(result.orig_img, channels="BGR")
        ann_frame.image(result.plot(), channels="BGR")
        

if __name__=="__main__":
    # Streamlit UI
    st.title("Object Detection Demo")

    model_selection = st.sidebar.selectbox(
        "Model",
        ("yolov11_out_of_the_box", "yolov11_aircraft_detect"),
    )
    if model_selection == "yolov11_aircraft_detect":
        model_path = "./runs/detect/train/weights/best.onnx"
        model = YOLO(model_path)
        class_names = list(model.names.values())  
        selected_classes = st.sidebar.multiselect("Classes", ["all"] + class_names, default="all")
        if selected_classes == ["all"]:
            selected_ind = list(range(0, len(class_names)))
        else:
            selected_ind = [class_names.index(option) for option in selected_classes]
    elif model_selection == "yolov11_out_of_the_box":
        model_path = "yolo11m.onnx"
        model = YOLO(model_path)
        class_names = list(model.names.values())  
        selected_classes = st.sidebar.multiselect("Classes", ["all"] + class_names, default="all")
        if selected_classes == ["all"]:
            selected_ind = list(range(0, len(class_names)))
        else:
            selected_ind = [class_names.index(option) for option in selected_classes]

    source = st.sidebar.selectbox(
        "Source",
        ("webcam", "video", "image"),
    )

    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

    if source == "video":
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file:
            with st.spinner("Processing video..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                    temp_file.write(vid_file.read())
                    video_path = temp_file.name
    elif source == "webcam":
        video_path = 0

    elif source == "image":
        img_file = st.sidebar.file_uploader("Upload Image File", type=["jpeg", "jpg", "png"])
        if img_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(img_file.read())
                img_path = temp_file.name

    if st.button('Start Demo'):
        col1, col2 = st.columns(2)
        org_frame = col1.empty()
        ann_frame = col2.empty()

        if source in ["video", "webcam"]:
            process_video(video_path=video_path, model=model, selected_ind=selected_ind)

        elif source == "image":
            process_image(img_path, model, selected_ind)
        


    # # File uploader
    # onnx_path = "./runs/detect/train/weights/best.onnx"#= st.text_input("Path to YOLO ONNX Model", value="yolov4.onnx")

    # if uploaded_file and onnx_path:
    #     st.write("Processing video...")
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
    #         temp_file.write(uploaded_file.read())
    #         video_path = temp_file.name

    #     # Load model
    #     yolo_model = load_model(onnx_path)
        
    #     # Process video
    #     input_shape = (640, 640)  # Adjust based on your YOLO model
    #     detections = process_video(video_path, yolo_model, input_shape)

    #     # Display results
    #     st.write(f"Total segments with objects: {len(detections)}")
    #     for detection in detections:
    #         frame_no = detection["frame"]
    #         classes_detected = detection["classes"]
    #         st.write(f"Frame: {frame_no}, Classes: {classes_detected}")

    #     os.remove(video_path)
