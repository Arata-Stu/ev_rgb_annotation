import numpy as np
import cv2
from omegaconf import OmegaConf
import argparse
import yaml

# 1. キャリブレーションデータの読み込み
def load_calibration_data(yaml_file_path):
    config = OmegaConf.load(yaml_file_path)
    return config

def create_intrinsic_matrix(intrinsics):
    fx, fy, cx, cy = intrinsics
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return K

def get_distortion_coeffs(distortion_coeffs):
    return np.array(distortion_coeffs)

def initialize_calibration(yaml_file_path):
    config = load_calibration_data(yaml_file_path)
    cameras = {}
    for cam_key in config.keys():
        cam_data = config[cam_key]
        K = create_intrinsic_matrix(cam_data.intrinsics)
        distortion = get_distortion_coeffs(cam_data.distortion_coeffs)
        cameras[cam_key] = {"K": K, "distortion": distortion}
    return cameras

# 2. ホモグラフィ行列の計算
def compute_homography(src_points, dst_points):
    H, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    return H

def apply_homography(H, points):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points_h = np.dot(H, points_h.T).T
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2][:, np.newaxis]
    return transformed_points

# 3. 対応点の手動選択機能
def select_points_dual_view(rgb_image, event_image):
    rgb_points = []
    event_points = []
    rgb_point_index = 1
    event_point_index = 1

    def mouse_callback_rgb(event, x, y, flags, param):
        nonlocal rgb_point_index
        if event == cv2.EVENT_LBUTTONDOWN:
            rgb_points.append((x, y))
            label = f"RGB-{rgb_point_index}"
            rgb_point_index += 1
            cv2.circle(rgb_image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(rgb_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("RGB Image", rgb_image)

    def mouse_callback_event(event, x, y, flags, param):
        nonlocal event_point_index
        if event == cv2.EVENT_LBUTTONDOWN:
            event_points.append((x, y))
            label = f"Event-{event_point_index}"
            event_point_index += 1
            cv2.circle(event_image, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(event_image, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Event Image", event_image)

    print("Click on points in the RGB image and Event image sequentially. Press 'q' when done.")

    cv2.imshow("RGB Image", rgb_image)
    cv2.imshow("Event Image", event_image)

    cv2.setMouseCallback("RGB Image", mouse_callback_rgb)
    cv2.setMouseCallback("Event Image", mouse_callback_event)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(rgb_points, dtype=np.float32), np.array(event_points, dtype=np.float32)

# 4. BBoxをホモグラフィで変換
def transform_bbox_with_homography(H, bbox):
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

# 5. ホモグラフィ行列をYAMLファイルに保存
def save_homography_to_yaml(H, output_file_path):
    with open(output_file_path, 'w') as file:
        yaml.dump({'homography': H.tolist()}, file, default_flow_style=False)

# 6. プロセス全体の実行例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform bounding box using homography.")
    parser.add_argument("--rgb_image", required=True, help="Path to the RGB image.")
    parser.add_argument("--event_image", required=True, help="Path to the Event image.")
    parser.add_argument("--calibration_file", required=True, help="Path to the calibration file (YAML).")

    args = parser.parse_args()

    # キャリブレーションデータのロード
    calibration_data = initialize_calibration(args.calibration_file)

    # RGBカメラ画像とイベントカメラ画像の読み込み
    rgb_image = cv2.imread(args.rgb_image)
    event_image = cv2.imread(args.event_image)

    if rgb_image is None or event_image is None:
        raise FileNotFoundError("One or both of the input images could not be loaded.")

    # 対応点の手動選択
    src_points, dst_points = select_points_dual_view(rgb_image.copy(), event_image.copy())

    # ホモグラフィ行列の計算と保存
    if len(src_points) == len(dst_points) and len(src_points) >= 4:
        H = compute_homography(src_points, dst_points)
        print("Homography Matrix:")
        print(H)

        # ホモグラフィ行列をYAMLに保存
        output_yaml_path = "homography_matrix.yaml"
        save_homography_to_yaml(H, output_yaml_path)
        print(f"Homography matrix saved to {output_yaml_path}")

        # RGB画像でのBBox選択
        bbox = cv2.selectROI("RGB Image", rgb_image, fromCenter=False, showCrosshair=True)
        x_min, y_min, width, height = bbox
        bbox = (x_min, y_min, x_min + width, y_min + height)
        print("Selected BBox:", bbox)

        # BBoxを変換
        transformed_bbox = transform_bbox_with_homography(H, bbox)
        print("Transformed BBox:", transformed_bbox)

        # イベント画像に描画
        cv2.rectangle(event_image, (transformed_bbox[0], transformed_bbox[1]),
                      (transformed_bbox[2], transformed_bbox[3]), (0, 255, 0), 2)
        cv2.imshow("Event Image with Transformed BBox", event_image)
        cv2.waitKey(0)
    else:
        print("Error: The number of points in RGB and Event images must match and be at least 4.")

    cv2.destroyAllWindows()
