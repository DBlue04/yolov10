import streamlit as st
from PIL import Image
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import supervision as sv
from ultralytics import YOLOv10


with open("evaluation.txt", "r") as f:
    evaluation_metrics = f.read()


st.write("### Model Evaluation Metrics")
st.text(evaluation_metrics)

model = YOLOv10('runs/detect/train/weights/best.pt')

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def process_and_display(img, is_video=False):
    results = model(source=img, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_img = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections)

    return annotated_img

upload_type = st.radio("Choose input type:", ("Upload Video", "Live Track"))

if upload_type == "Upload Video":
    uploaded_file = st.file_uploader(label='Upload video here', type=['mp4', 'mov'])
    if uploaded_file:
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        vid = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            annotated_frame = process_and_display(frame, is_video=True)
            stframe.image(annotated_frame)

        vid.release()

elif upload_type == "Live Track":
    st.write("### Live Object Tracking with YOLOv10")

    st.write("Starting webcam for live object tracking...")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not cap.isOpened():
        st.write("Error: Could not access the webcam.")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Failed to capture frame from webcam.")
                break

            annotated_frame = process_and_display(frame)
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        cap.release()
