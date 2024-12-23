import streamlit as st
import cv2
import io
import tempfile
from ultralytics import YOLO
import time

def process_video_sequenced(video_path, model, selected_ind):
    videocapture = cv2.VideoCapture(video_path)  
    if not videocapture.isOpened():
        st.error("Could not open webcam.")
    fps = int(videocapture.get(cv2.CAP_PROP_FPS))
    fps = fps * 0.5 if half_fr else fps
    n_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_text = "Running model on video"
    progress_bar = st.progress(0, text=progress_text)

    stop_button = st.button("Stop Demo") 

    original_frames = []
    annotated_frames = []
    count=0

    while count < n_frames:
        if (not half_fr) or (count % 2==0):
            success, frame = videocapture.read()
            if not success: break
            results = model(frame, conf=conf, iou=iou, classes=selected_ind)
            original_frames.append(frame)
            annotated_frames.append(results[0].plot()) 
        count+=1
        percent_complete = int(100*(count/n_frames))
        progress_bar.progress(percent_complete, text=f"{progress_text} ({percent_complete}%)")

        if stop_button:
            videocapture.release()  
            cv2.destroyAllWindows()
            st.stop()  
        
    progress_bar.empty()
    while True:
        for i in range(len(annotated_frames)-1):
            org_frame.image(original_frames[i], channels="BGR")
            ann_frame.image(annotated_frames[i], channels="BGR")
            time.sleep(1/fps)

            if stop_button:
                videocapture.release()  
                cv2.destroyAllWindows()
                st.stop()  


def process_video(video_path, model, selected_ind):
    videocapture = cv2.VideoCapture(video_path)  
    if not videocapture.isOpened():
        st.error("Could not open webcam.")

    fps_display = st.sidebar.empty() 
    stop_button = st.button("Stop Demo")  
    count = 0
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success: break
        if (not half_fr) or (count % 2==0):
            prev_time = time.time()  
            results = model(frame, conf=conf, iou=iou, classes=selected_ind)
            annotated_frame = results[0].plot()  
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)

            org_frame.image(frame, channels="BGR")
            ann_frame.image(annotated_frame, channels="BGR")

            fps_display.metric("FPS", f"{fps:.2f}")
        count+=1
        if stop_button:
            videocapture.release()  
            cv2.destroyAllWindows()
            st.stop()  

def process_webcam(video_path, model, selected_ind):
    videocapture = cv2.VideoCapture(video_path)  
    if not videocapture.isOpened():
        st.error("Could not open webcam.")

    fps_display = st.sidebar.empty() 
    stop_button = st.button("Stop Demo")  
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success: break
        
        prev_time = time.time()  
        results = model(frame, conf=conf, iou=iou, classes=selected_ind)
        annotated_frame = results[0].plot()  
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)

        org_frame.image(frame, channels="BGR")
        ann_frame.image(annotated_frame, channels="BGR")

        fps_display.metric("FPS", f"{fps:.2f}")

        if stop_button:
            videocapture.release()  
            cv2.destroyAllWindows()
            st.stop()  

def process_image(image_path, model, selected_ind):
    results = model([image_path], conf=conf, iou=iou, classes=selected_ind)
    # Process results list
    for result in results:
        org_frame.image(result.orig_img, channels="BGR")
        ann_frame.image(result.plot(), channels="BGR")
        

if __name__=="__main__":
    st.title("Aircraft Detection Demo")

    model_selection = st.sidebar.selectbox(
        "Model",
        ("nano", "small", "medium"),
    )
    suffix_map = {
        "nano":"n",
        "small":"s",
        "medium":"m"
    }
    model_path = f"./assets/best{suffix_map[model_selection]}.pt"
    model = YOLO(model_path)
    class_names = list(model.names.values())  

    selected_classes = st.sidebar.multiselect("Classes", ["all"] + class_names, default="all")
    if "all" in selected_classes:
        selected_ind = list(range(0, len(class_names)))
    else:
        selected_ind = [class_names.index(option) for option in selected_classes]

    source = st.sidebar.selectbox(
        "Source",
        #("webcam", "video", "image"),
        ("video", "image"),
    )

    conf = float(st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01))
    iou = float(st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01))

    if source == "image":
        input_file = st.sidebar.file_uploader("Upload Image File", type=["jpeg", "jpg", "png"])
        if input_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as temp_file:
                temp_file.write(input_file.read())
                input_file_path = temp_file.name
    elif source == "video":
        input_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if input_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(input_file.read())
                input_file_path = temp_file.name
        # if input_file is not None:
        #     g = io.BytesIO(input_file.read())  # BytesIO Object
        #     vid_location = "upload.mp4"
        #     with open(vid_location, "wb") as out:  # Open temporary file as bytes
        #         out.write(g.read())  # Read bytes into file
        #     input_file_path = "upload.mp4"
    
        half_fr = st.sidebar.checkbox("Reduce input frame rate by 0.5x (recommended)", value=True)


    elif source == "webcam":
        input_file_path = 0

    if st.button('Start Demo'):
        col1, col2 = st.columns(2)
        org_frame = col1.empty()
        ann_frame = col2.empty()
        if source == "webcam":
            process_webcam(video_path=input_file_path, model=model, selected_ind=selected_ind)
        elif source == "video":
            process_webcam(video_path=input_file_path, model=model, selected_ind=selected_ind)
        elif source == "image":
            process_image(input_file_path, model, selected_ind)
    
    
        
