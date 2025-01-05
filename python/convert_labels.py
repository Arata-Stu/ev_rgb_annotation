import os
import numpy as np
import argparse
import yaml

def load_homography(matrix_path, camera_name):
    """
    Load a homography matrix for a specific camera from a YAML file.

    Parameters:
        matrix_path (str): Path to the YAML file.
        camera_name (str): Name of the camera to load the matrix for.

    Returns:
        np.ndarray: Homography matrix (3x3) for the specified camera.
    """
    with open(matrix_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if camera_name not in data['homography_matrix']:
        raise ValueError(f"Camera {camera_name} not found in the homography matrix file.")

    return np.array(data['homography_matrix'][camera_name])

def apply_homography(H, points):
    """
    Apply a homography transformation to a set of points.

    Parameters:
        H (np.ndarray): Homography matrix (3x3).
        points (np.ndarray): Array of points with shape (N, 2).

    Returns:
        transformed_points (np.ndarray): Transformed points with shape (N, 2).
    """
    num_points = points.shape[0]
    homogenous_points = np.hstack([points, np.ones((num_points, 1))])  # (x, y) -> (x, y, 1)
    transformed_points = (H @ homogenous_points.T).T  # Apply homography
    transformed_points /= transformed_points[:, 2][:, None]  # Normalize to (x, y)
    return transformed_points[:, :2]  # Return only (x, y)

def transform_bbox_with_homography(H, bbox):
    """
    Transform a bounding box using a homography matrix.

    Parameters:
        H (np.ndarray): Homography matrix (3x3).
        bbox (tuple): Bounding box in the format (x, y, w, h).

    Returns:
        transformed_bbox (tuple): Transformed bounding box in the format (x, y, w, h).
    """
    x_center, y_center, w, h = bbox
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2

    corners = np.array([
        [x_min, y_min],  # Top-left
        [x_max, y_min],  # Top-right
        [x_max, y_max],  # Bottom-right
        [x_min, y_max]   # Bottom-left
    ])
    transformed_corners = apply_homography(H, corners)
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)

    new_x_center = (x_min + x_max) / 2
    new_y_center = (y_min + y_max) / 2
    new_w = x_max - x_min
    new_h = y_max - y_min

    return new_x_center, new_y_center, new_w, new_h

def transform_labels(base_dir, matrix_path):
    """
    ラベルの座標をホモグラフィ行列を使って別の座標系に変換し、結果を保存します。

    Args:
        base_dir (str): ベースディレクトリのパス。
        matrix_path (str): ホモグラフィ行列を保存したYAMLファイルのパス。
    """
    labels_dir = os.path.join(base_dir, "labels")

    camera_dirs = [
        d for d in os.listdir(labels_dir)
        if os.path.isdir(os.path.join(labels_dir, d)) and "camera" in d
    ]

    if not camera_dirs:
        print("Error: No camera directories found in the labels directory.")
        return

    for camera_dir in camera_dirs:
        input_dir = os.path.join(labels_dir, camera_dir)
        output_dir = os.path.join(labels_dir, f"events_{camera_dir}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            homography_matrix = load_homography(matrix_path, camera_dir)
        except ValueError as e:
            print(e)
            continue

        label_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])

        for label_file in label_files:
            label_path = os.path.join(input_dir, label_file)
            labels = np.load(label_path)

            transformed_labels = []

            for label in labels:
                x, y, w, h, class_id, confidence, track_id = label

                # バウンディングボックスをホモグラフィ行列で変換
                transformed_bbox = transform_bbox_with_homography(
                    homography_matrix, (x, y, w, h)
                )

                # 新しいラベルを保存
                transformed_labels.append([
                    *transformed_bbox, class_id, confidence, track_id
                ])

            # numpy配列に変換して保存
            output_array = np.array(transformed_labels)
            output_file = os.path.join(output_dir, label_file)
            np.save(output_file, output_array)

            print(f"Transformed {label_file} -> {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ラベルをホモグラフィ行列を使って別の座標系に変換します。")
    parser.add_argument("-b", "--base_dir", required=True, help="ベースディレクトリへのパス")
    parser.add_argument("-m", "--matrix", required=True, help="ホモグラフィ行列（YAMLファイル）へのパス")

    args = parser.parse_args()

    transform_labels(args.base_dir, args.matrix)
