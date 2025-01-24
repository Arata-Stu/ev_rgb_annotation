import os
import numpy as np
import argparse
import yaml
from tqdm import tqdm

def load_homography(matrix_path, camera_name):
    """YAMLファイルからホモグラフィ行列を読み込む"""
    with open(matrix_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if camera_name not in data['homography_matrix']:
        raise ValueError(f"Camera {camera_name} not found in the homography matrix file.")

    return np.array(data['homography_matrix'][camera_name])

def apply_homography(H, points):
    """ホモグラフィ行列を適用して点を変換"""
    num_points = points.shape[0]
    homogenous_points = np.hstack([points, np.ones((num_points, 1))])
    transformed_points = (H @ homogenous_points.T).T
    transformed_points /= transformed_points[:, 2][:, None]
    return transformed_points[:, :2]

def transform_bbox_with_homography(H, bbox):
    """バウンディングボックスをホモグラフィ変換"""
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

def process_labels(base_dir, matrix_path):
    """YOLOの推論結果をホモグラフィ変換し、全カメラ統合して保存"""
    labels_dir = os.path.join(base_dir, "labels")
    output_file = os.path.join(labels_dir, "labels_events.npy")
    
    if not os.path.exists(labels_dir):
        print("Error: labels directory not found.")
        return
    
    label_files = [
        f for f in os.listdir(labels_dir)
        if f.endswith('.npy') and "camera" in f
    ]
    
    if not label_files:
        print("Error: No label files found in the labels directory.")
        return
    
    all_transformed_data = []

    for label_file in label_files:
        camera_name = label_file.replace(".npy", "")
        input_file = os.path.join(labels_dir, label_file)

        try:
            homography_matrix = load_homography(matrix_path, camera_name)
        except ValueError as e:
            print(e)
            continue
        
        print(f"Processing {label_file} with homography transformation.")

        # npyファイルのロード
        data = np.load(input_file)

        if data.size == 0:
            print(f"Warning: {label_file} is empty. Skipping.")
            continue

        transformed_data = []
        
        for entry in tqdm(data, desc=f"Transforming {label_file}"):
            timestamp, x, y, w, h, class_id, conf, track_id = entry
            transformed_bbox = transform_bbox_with_homography(homography_matrix, (x, y, w, h))
            transformed_data.append([
                timestamp, *transformed_bbox, class_id, conf, track_id, camera_name
            ])
        
        all_transformed_data.extend(transformed_data)

    # 変換データを1つの npy ファイルに保存
    np.save(output_file, np.array(all_transformed_data, dtype=object))
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOの推論結果をホモグラフィ変換し、全カメラのデータを統合して保存します。"
    )
    parser.add_argument("-b", "--base_dir", required=True, help="ベースディレクトリへのパス")
    parser.add_argument("-m", "--matrix", required=True, help="ホモグラフィ行列（YAMLファイル）へのパス")

    args = parser.parse_args()
    process_labels(args.base_dir, args.matrix)
