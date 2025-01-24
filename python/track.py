import os
import numpy as np
import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm

def process_images(base_dir, target_classes):
    """
    指定ディレクトリ内のRGB画像にByteTrackを用いたYOLOを適用し、
    ラベルおよびTrackIDを1つの.npyファイルに統合する。
    
    Args:
        base_dir (str): ベースディレクトリのパス。
        target_classes (list): 検出対象のクラス名リスト。
    """
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")
    os.makedirs(labels_dir, exist_ok=True)
    
    # image_offsets.txt の読み込み
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
        output_file = os.path.join(labels_dir, f"{camera_dir}.npy")  # 修正: labels/cameraX.npy に保存
        
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
                            output_data.append([
                                timestamp,
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                                int(cls_id),
                                float(conf),
                                int(track_id)
                            ])
            
            all_data.extend(output_data)
        
        np.save(output_file, np.array(all_data, dtype=np.float32))  # 修正: labels/cameraX.npy に保存
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOを用いて画像からラベルとトラッキングIDを抽出し、1つのnpyファイルとして保存します。"
    )
    parser.add_argument("-b", "--base_dir", required=True, help="ベースディレクトリへのパス")
    parser.add_argument(
        "-c", "--classes",
        nargs='*',
        default=["car", "bicycle", "person", "motorcycle", "bus"],
        help="検出対象のクラス名リスト（スペース区切り）。ALLを指定するとすべてのクラスを対象にします。"
    )

    args = parser.parse_args()
    process_images(args.base_dir, args.classes)
