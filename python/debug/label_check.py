import os
import cv2
import numpy as np
import argparse
import random

def visualize_labels(image_dir, label_dir, match_mode="filename"):
    """
    画像と対応するラベル（.npyファイル）を可視化し、前後に移動可能なインターフェースを提供。

    Args:
        image_dir (str): 入力画像が格納されたディレクトリパス。
        label_dir (str): ラベル（.npyファイル）が格納されたディレクトリパス。
        match_mode (str): ファイルの対応モード ("filename" または "index")。
    """

    # ここでトラッキングIDごとのカラーを保持する辞書を作成
    track_colors = {}

    # ディレクトリ内の画像ファイルとラベルファイルを取得し、ソート
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

    # 検出されたファイル数を表示
    print(f"Found {len(image_files)} image files in '{image_dir}'.")
    print(f"Found {len(label_files)} label files in '{label_dir}'.")

    if match_mode not in ["filename", "index"]:
        print("Invalid match_mode. Please use 'filename' or 'index'.")
        return

    idx = 0  # 初期表示のインデックス
    while 0 <= idx < len(image_files):
        if match_mode == "filename":
            # ファイル名で対応
            image_file = image_files[idx]
            label_file = os.path.splitext(image_file)[0] + ".npy"
        elif match_mode == "index":
            # インデックスで対応
            image_file = image_files[idx]
            label_file = label_files[idx]

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file not found for image: {image_file}")
            idx += 1  # 次の画像に進む
            continue

        # 画像とラベルを表示
        should_exit = display_image_with_labels(image_path, label_path, track_colors)
        if should_exit:
            break

        # キー入力で前後の画像に移動
        key = cv2.waitKey(0)  # キー入力待機
        if key == ord('q'):  # 'q' を押したら終了
            break
        elif key == ord('n'):  # 'n' で次の画像
            idx += 1
        elif key == ord('p'):  # 'p' で前の画像
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()

def display_image_with_labels(image_path, label_path, track_colors):
    """
    画像とラベルを読み込み、バウンディングボックスを描画して表示。

    Args:
        image_path (str): 画像ファイルのパス。
        label_path (str): ラベルファイルのパス。
        track_colors (dict): トラッキングIDごとのカラーを保持する辞書。
    Returns:
        bool: 終了フラグ（Trueの場合終了）。
    """

    # すでに定義済みのtrack_colorsを利用して色を取得・保持する
    def get_color(track_id):
        if track_id not in track_colors:
            track_colors[track_id] = [random.randint(0, 255) for _ in range(3)]
        return track_colors[track_id]

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return False

    labels = np.load(label_path)

    for label in labels:
        x1, y1, w, h, class_id, confidence, track_id = label
        x1, y1, w, h = map(int, [x1, y1, w, h])

        color = get_color(int(track_id))  # 同じtrack_idなら常に同じ色を返す
        x2, y2 = x1 + w, y1 + h

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID: {int(track_id)} Class: {int(class_id)} Conf: {confidence:.2f}",
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
    parser.add_argument("-m", "--match_mode", choices=["filename", "index"], default="filename",
                        help="ファイル対応モード（デフォルト: filename）")

    args = parser.parse_args()
    visualize_labels(args.image_dir, args.label_dir, args.match_mode)
