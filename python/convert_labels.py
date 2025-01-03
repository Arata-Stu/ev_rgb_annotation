import os
import numpy as np
import argparse
import yaml

def load_homography(matrix_path):
    """
    Load a homography matrix from a file.

    Parameters:
        matrix_path (str): Path to the matrix file (YAML or NPY).

    Returns:
        np.ndarray: Homography matrix (3x3).
    """
    if matrix_path.endswith('.npy'):
        # NPYファイルの場合
        return np.load(matrix_path)
    elif matrix_path.endswith('.yaml') or matrix_path.endswith('.yml'):
        # YAMLファイルの場合
        with open(matrix_path, 'r') as f:
            data = yaml.safe_load(f)
        return np.array(data['homography_matrix'])
    else:
        raise ValueError("Unsupported file format. Use '.npy' or '.yaml'.")

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
        bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).

    Returns:
        transformed_bbox (tuple): Transformed bounding box (x_min, y_min, x_max, y_max).
    """
    corners = np.array([
        [bbox[0], bbox[1]],  # Top-left
        [bbox[2], bbox[1]],  # Top-right
        [bbox[2], bbox[3]],  # Bottom-right
        [bbox[0], bbox[3]]   # Bottom-left
    ])
    transformed_corners = apply_homography(H, corners)
    x_min, y_min = np.min(transformed_corners, axis=0)
    x_max, y_max = np.max(transformed_corners, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

def transform_labels(label_dir, homography_matrix, output_dir):
    """
    ラベルの座標をホモグラフィ行列を使って別の座標系に変換し、結果を保存します。

    Args:
        label_dir (str): 入力ラベル（.npyファイル）が格納されたディレクトリパス。
        homography_matrix (np.ndarray): 3x3のホモグラフィ行列。
        output_dir (str): 変換後のラベルを保存するディレクトリパス。
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # ラベルファイルを取得
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        labels = np.load(label_path)

        transformed_labels = []

        for label in labels:
            x_min, y_min, x_max, y_max, class_id, confidence = label

            # バウンディングボックスをホモグラフィ行列で変換
            transformed_bbox = transform_bbox_with_homography(
                homography_matrix, (x_min, y_min, x_max, y_max)
            )

            # 新しいラベルを保存
            transformed_labels.append([
                *transformed_bbox, class_id, confidence
            ])

        # numpy配列に変換して保存
        output_array = np.array(transformed_labels)
        output_file = os.path.join(output_dir, label_file)
        np.save(output_file, output_array)

        print(f"Transformed {label_file} -> {output_file}")

if __name__ == "__main__":
    # コマンドライン引数のパーサを設定
    parser = argparse.ArgumentParser(description="ラベルをホモグラフィ行列を使って別の座標系に変換します。")
    parser.add_argument("-l", "--label_dir", required=True, help="入力ラベル（.npyファイル）が格納されたディレクトリへのパス")
    parser.add_argument("-m", "--matrix", required=True, help="ホモグラフィ行列（.npyまたは.yamlファイル）へのパス")
    parser.add_argument("-o", "--output_dir", required=True, help="変換後のラベルを保存するディレクトリへのパス")

    # 引数を解析
    args = parser.parse_args()

    # ホモグラフィ行列を読み込む
    homography_matrix = load_homography(args.matrix)

    # 関数を実行
    transform_labels(args.label_dir, homography_matrix, args.output_dir)
