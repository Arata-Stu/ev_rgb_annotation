import os
import numpy as np
import cv2
from ultralytics import YOLO
import argparse
from tqdm import tqdm

def process_images(base_dir, target_classes):
    """
    指定ディレクトリ内のRGB画像にByteTrackを用いたYOLOを適用し、
    ラベルおよびTrackIDを.npyファイルとして保存する。

    Args:
        base_dir (str): ベースディレクトリのパス。
        target_classes (list): 検出対象のクラス名リスト。
    """
    # imagesディレクトリとlabelsディレクトリのパスを構築
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels")

    # 出力ディレクトリを作成
    os.makedirs(labels_dir, exist_ok=True)

    # imagesディレクトリ内の"camera"を含むサブディレクトリを探索
    camera_dirs = [
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d)) and "camera" in d
    ]

    if not camera_dirs:
        print("Error: No camera directories found in the images directory.")
        return

    # YOLOモデルをロード（任意の重みファイルに置き換えてください）
    model = YOLO("yolo11x.pt")

    # 各カメラディレクトリを処理
    for camera_dir in camera_dirs:
        camera_path = os.path.join(images_dir, camera_dir)
        output_dir = os.path.join(labels_dir, camera_dir)
        os.makedirs(output_dir, exist_ok=True)

        # ディレクトリ内の画像ファイルを取得し、ソート
        image_files = sorted([
            f for f in os.listdir(camera_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        if not image_files:
            print(f"Warning: No image files found in the directory {camera_dir}.")
            continue

        print(f"Processing directory: {camera_dir}")

        # TQDMを使用して進行状況を表示
        for image_file in tqdm(image_files, desc=f"Processing {camera_dir}"):
            image_path = os.path.join(camera_path, image_file)

            # 画像を読み込む
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to load image: {image_file}")
                continue

            # YOLOモデルでトラッキング
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml"
            )

            # 画像1枚に対する全検出結果を格納するリスト
            output_data = []

            # トラッキング結果を処理
            if results is not None:
                for result in results:
                    boxes = result.boxes.xyxy  # [x1, y1, x2, y2]
                    confs = result.boxes.conf  # 信頼度
                    cls_ids = result.boxes.cls.int().cpu().tolist()  # クラスID
                    track_ids = result.boxes.id.cpu().tolist() if result.boxes.id is not None else [-1] * len(cls_ids)  # トラックID

                    for box, conf, cls_id, track_id in zip(boxes, confs, cls_ids, track_ids):
                        x1, y1, x2, y2 = box
                        class_name = model.names[cls_id]

                        # 対象クラスのみを処理
                        if not target_classes or class_name in target_classes:
                            output_data.append([
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                                int(cls_id),
                                float(conf),
                                int(track_id)
                            ])

            # numpy配列に変換して保存
            output_array = np.array(output_data, dtype=np.float32)
            output_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".npy")
            np.save(output_file, output_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOを用いて画像からラベルとトラッキングIDを抽出し、npyファイルとして保存します。"
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
