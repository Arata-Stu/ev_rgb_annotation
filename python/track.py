import os
import numpy as np
import cv2
from ultralytics import YOLO
import argparse

def process_images(input_dir, output_dir, target_classes):
    """
    指定ディレクトリ内のRGB画像にYOLOを適用し、ラベルを.npyファイルとして保存する。

    Args:
        input_dir (str): 入力画像が格納されたディレクトリパス。
        output_dir (str): 出力ラベルを保存するディレクトリパス。
        target_classes (list): 検出対象のクラス名リスト。
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # ディレクトリ内の画像ファイルを取得し、ソート（順序を保証）
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # YOLOモデルをロード
    model = YOLO("yolo11n.pt")

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)

        # 画像を読み込む
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_file}")
            continue

        # YOLOモデルで推論
        results = model.predict(frame)

        # 各画像の結果を格納するリスト
        output_data = []

        for result in results:
            for box in result.boxes:
                # クラス名を取得
                class_id = int(box.cls[0])
                class_name = model.names[class_id]  # モデルからクラス名を取得

                # 対象クラスのみを処理
                if not target_classes or class_name in target_classes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    confidence = float(box.conf[0])

                    # データを追加
                    output_data.append([x1, y1, x2, y2, class_id, confidence])

        # numpy配列に変換して保存
        output_array = np.array(output_data)
        output_file = os.path.join(output_dir, os.path.splitext(image_file)[0] + ".npy")
        np.save(output_file, output_array)

        print(f"Processed {image_file} -> {output_file}")

if __name__ == "__main__":
    # コマンドライン引数のパーサを設定
    parser = argparse.ArgumentParser(description="YOLOを用いて画像からラベルを抽出し、npyファイルとして保存します。")
    parser.add_argument("-i", "--input", required=True, help="入力ディレクトリへのパス")
    parser.add_argument("-o", "--output", required=True, help="出力ディレクトリへのパス")
    parser.add_argument(
        "-c", "--classes",
        nargs='+',
        default=["car", "bicycle", "person", "motorcycle", "bus"],  # デフォルトは空リスト（すべてのクラスを対象）
        help="検出対象のクラス名リスト（スペース区切り）。未指定の場合はすべてのクラスを対象とします。"
    )

    # 引数を解析
    args = parser.parse_args()

    # 関数を実行
    process_images(args.input, args.output, args.classes)
