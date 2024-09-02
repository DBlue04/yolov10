import streamlit as st
from PIL import Image
import cv2
from tempfile import NamedTemporaryFile
import numpy as np
import supervision as sv
from ultralytics import YOLOv10
from sklearn.metrics import precision_score, recall_score, f1_score

# Load your model
model = YOLOv10('runs/detect/train/weights/best.pt')

st.title('Object Detector')

upload_type = st.radio("Choose upload type:", ("Image", "Video"))

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

def process_and_display(img, ground_truth=None, is_video=False):
    results = model(source=img, conf=0.25)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_img = bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_img = label_annotator.annotate(scene=annotated_img, detections=detections)

    if ground_truth is not None:
        y_true = []
        y_pred = []

        for gt_box, gt_label in ground_truth:
            best_iou = 0
            best_pred_label = -1

            for pred_box, pred_conf, pred_label in zip(detections.xyxy, detections.confidence, detections.class_id):
                iou_value = iou(gt_box, pred_box)
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_pred_label = pred_label

            if best_iou >= 0.5:  # IoU threshold for a positive match
                y_true.append(gt_label)
                y_pred.append(best_pred_label)
            else:
                y_true.append(gt_label)
                y_pred.append(-1)  # No match found

        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        precision = recall = f1 = 0  # Dummy values for non-ground truth scenarios

    return annotated_img, precision, recall, f1

if upload_type == "Image":
    uploaded_file = st.file_uploader(label='Upload image here', type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_np = np.array(img)
        ground_truth = [((0, 0, 50, 50), 1), ((50, 50, 100, 100), 1)]  # Replace with actual ground truth
        annotated_img, precision, recall, f1 = process_and_display(img_np, ground_truth=ground_truth)
        st.image(annotated_img, caption="Detected Image", use_column_width=True)
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-Score: {f1:.2f}")

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
            ground_truth = [((0, 0, 50, 50), 1), ((50, 50, 100, 100), 1)]  # Replace with actual ground truth
            annotated_frame, precision, recall, f1 = process_and_display(frame, ground_truth=ground_truth, is_video=True)
            stframe.image(annotated_frame)
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1-Score: {f1:.2f}")

        vid.release()
