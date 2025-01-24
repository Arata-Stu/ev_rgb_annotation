import os
import numpy as np
import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm
import yaml

def load_homography(matrix_path, camera_name):
    with open(matrix_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if camera_name not in data['homography_matrix']:
        raise ValueError(f"Camera {camera_name} not found in the homography matrix file.")

    return np.array(data['homography_matrix'][camera_name])

def apply_homography(H, points):
    num_points = points.shape[0]
    homogenous_points = np.hstack([points, np.ones((num_points, 1))])
    transformed_points = (H @ homogenous_points.T).T
    transformed_points /= transformed_points[:, 2][:, None]
    return transformed_points[:, :2]

def transform_bbox_with_homography(H, bbox):
    x_center, y_center, w, h = bbox
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    transformed_corners = apply_homography(H, corners)
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)

    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2
    new_w = x_max - x_min
    new_h = y_max - y_min

    return new_x_center, new_y_center, new_w, new_h

def process_images(base_dir, target_classes, matrix_path):
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    offset_file = os.path.join(base_dir, "image_offsets.txt")
    if not os.path.exists(offset_file):
        print("Error: image_offsets.txt not found.")
        return
    
    with open(offset_file, "r") as f:
        timestamps = [int(line.strip()) for line in f.readlines()]
    
    camera_dirs = [
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d)) and "camera" in d
    ]
    
    if not camera_dirs:
        print("Error: No camera directories found in the images directory.")
        return
    
    model = YOLO("yolo11x.pt")
    
    for camera_dir in camera_dirs:
        camera_path = os.path.join(images_dir, camera_dir)
        camera_label_dir = os.path.join(labels_dir, camera_dir)
        os.makedirs(camera_label_dir, exist_ok=True)
        output_file = os.path.join(camera_label_dir, f"{camera_dir}.npy")
        
        try:
            homography_matrix = load_homography(matrix_path, camera_dir)
        except ValueError as e:
            print(e)
            continue
        
        image_files = sorted([
            f for f in os.listdir(camera_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not image_files:
            print(f"Warning: No image files found in the directory {camera_dir}.")
            continue
        
        print(f"Processing directory: {camera_dir}")
        
        all_data = []
        
        for idx, image_file in enumerate(tqdm(image_files, desc=f"Processing {camera_dir}")):
            image_path = os.path.join(camera_path, image_file)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to load image: {image_file}")
                continue
            
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml"
            )
            
            timestamp = timestamps[idx] if idx < len(timestamps) else -1
            output_data = []
            
            if results is not None:
                for result in results:
                    boxes = result.boxes.xyxy
                    confs = result.boxes.conf
                    cls_ids = result.boxes.cls.int().cpu().tolist()
                    track_ids = result.boxes.id.cpu().tolist() if result.boxes.id is not None else [-1] * len(cls_ids)
                    
                    for box, conf, cls_id, track_id in zip(boxes, confs, cls_ids, track_ids):
                        x1, y1, x2, y2 = box
                        class_name = model.names[cls_id]
                        
                        if not target_classes or class_name in target_classes:
                            bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                            transformed_bbox = transform_bbox_with_homography(homography_matrix, bbox)
                            output_data.append([
                                timestamp,
                                *transformed_bbox, int(cls_id), float(conf), int(track_id)
                            ])
            
            all_data.extend(output_data)
        
        np.save(output_file, np.array(all_data, dtype=np.float32))
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOを用いて画像からラベルとトラッキングIDを抽出し、ホモグラフィ変換後の座標系に変換して保存します。"
    )
    parser.add_argument("-b", "--base_dir", required=True, help="ベースディレクトリへのパス")
    parser.add_argument("-c", "--classes", nargs='*', default=["car", "bicycle", "person", "motorcycle", "bus"], help="検出対象のクラス名リスト（スペース区切り）。")
    parser.add_argument("-m", "--matrix", required=True, help="ホモグラフィ行列（YAMLファイル）へのパス")

    args = parser.parse_args()
    process_images(args.base_dir, args.classes, args.matrix)
