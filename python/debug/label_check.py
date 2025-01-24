import os
import cv2
import numpy as np
import argparse
import random

def visualize_labels(image_dir, label_dir):
    track_colors = {}
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {len(image_files)} image files in '{image_dir}'.")
    
    label_file = os.path.join(label_dir, f"{os.path.basename(os.path.normpath(label_dir))}.npy")
    if not os.path.exists(label_file):
        print(f"Label file not found: {label_file}")
        return
    
    labels = np.load(label_file)
    unique_timestamps = sorted(set(labels[:, 0]))
    
    print(f"Found {len(unique_timestamps)} unique timestamps in label file.")
    
    if len(image_files) != len(unique_timestamps):
        print("Warning: Number of image files and unique labels do not match.")
    
    idx = 0
    while 0 <= idx < min(len(image_files), len(unique_timestamps)):
        image_file = image_files[idx]
        timestamp = unique_timestamps[idx]  # インデックスで対応
        
        image_path = os.path.join(image_dir, image_file)
        should_exit = display_image_with_labels(image_path, label_file, timestamp, track_colors)
        if should_exit:
            break
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == ord('n'):
            idx += 1
        elif key == ord('p'):
            idx = max(0, idx - 1)
    
    cv2.destroyAllWindows()

def display_image_with_labels(image_path, label_path, timestamp, track_colors):
    def get_color(track_id):
        if track_id not in track_colors:
            track_colors[track_id] = [random.randint(0, 255) for _ in range(3)]
        return track_colors[track_id]
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    labels = np.load(label_path)
    frame_labels = labels[labels[:, 0] == timestamp]  # 指定タイムスタンプのデータのみ取得
    
    for label in frame_labels:
        _, x1, y1, w, h, class_id, confidence, track_id = label
        x1, y1, w, h = map(int, [x1, y1, w, h])
        
        color = get_color(int(track_id))
        x2, y2 = x1 + w, y1 + h
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID: {int(track_id)} Class: {int(class_id)} Conf: {confidence:.2f} Time: {timestamp}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    cv2.imshow("Image with Labels", frame)
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像と対応するラベル（.npyファイル）を可視化します。")
    parser.add_argument("-i", "--image_dir", required=True, help="入力画像が格納されたディレクトリへのパス")
    parser.add_argument("-l", "--label_dir", required=True, help="ラベル（.npyファイル）が格納されたディレクトリへのパス")
    
    args = parser.parse_args()
    visualize_labels(args.image_dir, args.label_dir)
