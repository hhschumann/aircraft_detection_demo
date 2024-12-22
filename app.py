import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time

def process_video(video_path, model, selected_ind):
    videocapture = cv2.VideoCapture(video_path)  
    if not videocapture.isOpened():
        st.error("Could not open webcam.")

    fps_display = st.sidebar.empty() 
    stop_button = st.button("Stop Demo")  
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            st.warning("Failed to read frame from webcam...")
            break

        prev_time = time.time()  
        results = model(frame, conf=conf, iou=iou, classes=selected_ind)
        annotated_frame = results[0].plot()  
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)

        org_frame.image(frame, channels="BGR")
        ann_frame.image(annotated_frame, channels="BGR")

        if stop_button:
            videocapture.release()  
            st.stop()  

        fps_display.metric("FPS", f"{fps:.2f}")

    videocapture.release()
    cv2.destroyAllWindows()

def process_image(image_path, model, selected_ind):
    results = model([image_path], conf=conf, iou=iou, classes=selected_ind)
    # Process results list
    for result in results:
        org_frame.image(result.orig_img, channels="BGR")
        ann_frame.image(result.plot(), channels="BGR")
        

if __name__=="__main__":
    st.title("Object Detection Demo")

    model_selection = st.sidebar.selectbox(
        "Model",
        ("yolov11_aircraft_detect", "yolov11_out_of_the_box"),
    )

    model_size = st.sidebar.selectbox(
        "Model Size",
        ("nano", "small", "medium"),
    )

    if model_selection == "yolov11_aircraft_detect":
        model_path = f"./{model_size}/runs/detect/train/weights/best.onnx"
        model = YOLO(model_path)
        class_names = list(model.names.values())  

    elif model_selection == "yolov11_out_of_the_box":
        suffix_map = {
            "nano":"n",
            "small":"s",
            "medium":"m"
        }
        model_path = f"{model_size}/yolo11{suffix_map[model_size]}.onnx"
        model = YOLO(model_path)
        class_names = list(model.names.values())  

    selected_classes = st.sidebar.multiselect("Classes", ["all"] + class_names, default="all")
    if "all" in selected_classes:
        selected_ind = list(range(0, len(class_names)))
    else:
        selected_ind = [class_names.index(option) for option in selected_classes]

    source = st.sidebar.selectbox(
        "Source",
        ("webcam", "video", "image"),
    )

    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01))

    if source in ["video", "image"]:
        input_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if input_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(input_file.read())
                input_file_path = temp_file.name
    elif source == "webcam":
        input_file_path = 0

    if st.button('Start Demo'):
        col1, col2 = st.columns(2)
        org_frame = col1.empty()
        ann_frame = col2.empty()
        if source in ["video", "webcam"]:
            process_video(video_path=input_file_path, model=model, selected_ind=selected_ind)
        elif source == "image":
            process_image(input_file_path, model, selected_ind)
        
