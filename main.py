import streamlit as st
from PIL import Image
import cv2
from tempfile import NamedTemporaryFile
import requests
from io import BytesIO
from ultralytics import YOLOv10
import supervision as sv
import numpy as np
from pytube import YouTube

model = YOLOv10('runs/detect/train/weights/best.pt')

st.title('Object Detector')

# upload_option = st.radio("Choose an option:", ("Upload File", "Paste Link"))

def process_and_display(img):
    results = model(source=img, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_img = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections)

    # Calculate precision and recall
    precision = results.boxes.data['precision'].mean()  # Assuming results contain 'precision'
    recall = results.boxes.data['recall'].mean()  # Assuming results contain 'recall'

    return annotated_img, precision, recall

# if upload_option == "Upload File":
upload_type = st.radio("Choose upload type:", ("Image", "Video"))

if upload_type == "Image":
    uploaded_file = st.file_uploader(label='Upload image here', type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        annotated_img, precision, recall = process_and_display(img_np)
        st.image(annotated_img, caption="Detected Image", use_column_width=True)
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")

elif upload_type == "Video":
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
            annotated_frame, precision, recall = process_and_display(frame)
            stframe.image(annotated_frame)
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")

        vid.release()