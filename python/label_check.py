import os
import cv2
import numpy as np
import argparse

def visualize_labels(image_dir, label_dir):
    """
    画像と対応するラベル（.npyファイル）を可視化し、コマ送りで表示する。

    Args:
        image_dir (str): 入力画像が格納されたディレクトリパス。
        label_dir (str): ラベル（.npyファイル）が格納されたディレクトリパス。
    """
    # ディレクトリ内の画像ファイルを取得し、ソート（順序を保証）
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.npy')])

    # 検出されたファイル数を表示
    print(f"Found {len(image_files)} image files in '{image_dir}'.")
    print(f"Found {len(label_files)} label files in '{label_dir}'.")

    for image_file in image_files:
        # 対応するラベルファイルを取得
        label_file = os.path.splitext(image_file)[0] + ".npy"
        label_path = os.path.join(label_dir, label_file)

        # 画像を読み込む
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Failed to load image: {image_file}")
            continue

        # ラベルを読み込む
        if not os.path.exists(label_path):
            print(f"Label file not found for image: {image_file}")
            continue

        labels = np.load(label_path)

        # バウンディングボックスを描画
        for label in labels:
            x1, y1, x2, y2, class_id, confidence = label
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # バウンディングボックスとラベルを描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Class: {int(class_id)} Conf: {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # 画像を表示
        cv2.imshow("Image with Labels", frame)

        # コマ送り（キー入力で次の画像に進む）
        key = cv2.waitKey(0)  # 0: 無限待機
        if key == ord('q'):  # 'q' を押したら終了
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # コマンドライン引数のパーサを設定
    parser = argparse.ArgumentParser(description="画像と対応するラベル（.npyファイル）を可視化します。")
    parser.add_argument("-i", "--image_dir", required=True, help="入力画像が格納されたディレクトリへのパス")
    parser.add_argument("-l", "--label_dir", required=True, help="ラベル（.npyファイル）が格納されたディレクトリへのパス")

    # 引数を解析
    args = parser.parse_args()

    # 関数を実行
    visualize_labels(args.image_dir, args.label_dir)
