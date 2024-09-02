import numpy as np
import supervision as sv
from ultralytics import YOLOv10
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import os

model = YOLOv10('runs/detect/train/weights/best.pt')

test_dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=os.path.join(os.getcwd(), 'VehiclesDetectionDataset', 'test', 'images'),
    annotations_directory_path=os.path.join(os.getcwd(), 'VehiclesDetectionDataset', 'test', 'labels'),
    data_yaml_path='data.yaml'
)
# images_directory_path=os.path.join(os.getcwd(), 'yolov10', 'VehiclesDetectionDataset', 'test', 'images')
# print(images_directory_path)

# test_dataset1 = sv.DetectionDataset.from_yolo(
#     images_directory_path='/home/kelvin/my_project/fitus/Junior - Semester 3/code/yolov10/VehiclesDetectionDataset/test/images',
#     annotations_directory_path='/home/kelvin/my_project/fitus/Junior - Semester 3/code/yolov10/VehiclesDetectionDataset/test/labels',
#     data_yaml_path='data.yaml'
# )
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

# Evaluate model on test set
def evaluate_model_on_test_set():
    all_ground_truths = []
    all_predictions = []

    for image_path in tqdm(test_dataset.images.keys()):
        image = test_dataset.images[image_path]
        
        results = model(source=image, conf=0.25)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        ground_truth = test_dataset.annotations[image_path]
        predictions = np.array([
            [
                box[0],    # x_min
                box[1],    # y_min
                box[2],    # x_max
                box[3],    # y_max
                conf,      # confidence score
                cls_id     # class ID
            ]
            for box, conf, cls_id in zip(detections.xyxy, detections.confidence, detections.class_id)
        ])
        
        all_ground_truths.append(ground_truth)
        all_predictions.append(predictions)

    y_true = []
    y_pred = []

    for gt, pred in zip(all_ground_truths, all_predictions):
        gt_labels = gt.class_id
        gt_boxes = gt.xyxy
        
        if pred.size == 0:
            pred = np.empty((0, 6)) 

        if len(gt_labels) > 0:
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                best_iou = 0
                best_pred_label = -1 
                best_pred_index = -1

                for i, pred_box in enumerate(pred[:, :4]):
                    iou_value = iou(gt_box, pred_box)
                    if iou_value > best_iou:
                        best_iou = iou_value
                        best_pred_label = int(pred[i, 5])
                        best_pred_index = i
                
                if best_iou >= 0.5:
                    y_true.append(gt_label)
                    y_pred.append(best_pred_label)
                    
                    pred = np.delete(pred, best_pred_index, axis=0)
                else:
                    y_true.append(gt_label)
                    y_pred.append(-1)

            for remaining_pred in pred:
                y_true.append(-1)
                y_pred.append(int(remaining_pred[5]))

    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return precision, recall, f1

if __name__ == "__main__":
    precision, recall, f1 = evaluate_model_on_test_set()
    
    # Save evaluation metrics to a text file
    with open("evaluation.txt", "w") as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

    print("Model evaluation completed and saved to evaluation.txt")
