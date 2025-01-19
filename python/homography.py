import os
import numpy as np
import cv2
import yaml

# 1. ホモグラフィ行列の計算
def compute_homography(src_points, dst_points):
    H, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
    return H

def apply_homography(H, points):
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points_h = np.dot(H, points_h.T).T
    transformed_points = transformed_points_h[:, :2] / transformed_points_h[:, 2][:, np.newaxis]
    return transformed_points

# 2. 対応点の手動選択機能
def select_points_dual_view(image1, image2):
    points1 = []
    points2 = []
    point_index1 = 1
    point_index2 = 1

    def mouse_callback1(event, x, y, flags, param):
        nonlocal point_index1
        if event == cv2.EVENT_LBUTTONDOWN:
            points1.append((x, y))
            label = f"P1-{point_index1}"
            point_index1 += 1
            cv2.circle(image1, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image1, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Image 1", image1)

    def mouse_callback2(event, x, y, flags, param):
        nonlocal point_index2
        if event == cv2.EVENT_LBUTTONDOWN:
            points2.append((x, y))
            label = f"P2-{point_index2}"
            point_index2 += 1
            cv2.circle(image2, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(image2, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Image 2", image2)

    print("Click on points in Image 1 and Image 2 sequentially. Press 'q' when done.")

    cv2.imshow("Image 1", image1)
    cv2.imshow("Image 2", image2)

    cv2.setMouseCallback("Image 1", mouse_callback1)
    cv2.setMouseCallback("Image 2", mouse_callback2)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return np.array(points1, dtype=np.float32), np.array(points2, dtype=np.float32)

# 3. ホモグラフィ行列をYAMLファイルに保存
def save_homographies_to_yaml(homographies, output_file_path):
    with open(output_file_path, 'w') as file:
        yaml.dump({"homography_matrix": homographies}, file, default_flow_style=False)

# 4. プロセス全体の実行例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute homographies for images in structured directories.")
    parser.add_argument("-b", "--base_dir", required=True, help="Base directory containing image directories.")
    parser.add_argument("-o", "--output_yaml", default="./homography_matrix.yaml", help="Path to the output YAML file.")
    parser.add_argument("-f", "--frame_index", type=int, default=0, help="Index of the frame pair to process (default: 0).")

    args = parser.parse_args()

    base_dir = args.base_dir
    output_yaml = args.output_yaml
    frame_index = args.frame_index

    # ディレクトリの取得
    camera_dirs = [d for d in os.listdir(os.path.join(base_dir, "images")) if "camera" in d]
    event_dir = os.path.join(base_dir, "images", "events")

    if not os.path.isdir(event_dir):
        raise FileNotFoundError("Event directory not found.")

    event_images = sorted([os.path.join(event_dir, f) for f in os.listdir(event_dir) if f.endswith(('.png', '.jpg'))])

    if not event_images:
        raise FileNotFoundError("No event images found.")

    homographies = {}

    for camera_dir in camera_dirs:
        camera_path = os.path.join(base_dir, "images", camera_dir)
        rgb_images = sorted([os.path.join(camera_path, f) for f in os.listdir(camera_path) if f.endswith(('.png', '.jpg'))])

        if not rgb_images:
            print(f"No RGB images found in {camera_path}. Skipping.")
            continue

        if len(rgb_images) <= frame_index or len(event_images) <= frame_index:
            print(f"Frame index {frame_index} out of range for {camera_dir}. Skipping.")
            continue

        rgb_image_path = rgb_images[frame_index]
        event_image_path = event_images[frame_index]

        print(f"Processing frame index {frame_index}: {rgb_image_path} and {event_image_path}")

        rgb_image = cv2.imread(rgb_image_path)
        event_image = cv2.imread(event_image_path)

        if rgb_image is None or event_image is None:
            print(f"Error loading images: {rgb_image_path}, {event_image_path}. Skipping.")
            continue

        src_points, dst_points = select_points_dual_view(rgb_image.copy(), event_image.copy())

        if len(src_points) == len(dst_points) >= 4:
            H = compute_homography(src_points, dst_points)
            print(f"Homography Matrix for {camera_dir}:")
            print(H)

            homographies[camera_dir] = H.tolist()
        else:
            print(f"Insufficient points for {camera_dir}. Skipping.")

    save_homographies_to_yaml(homographies, output_yaml)
    print(f"All homography matrices saved to {output_yaml}")
