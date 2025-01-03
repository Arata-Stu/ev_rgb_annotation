import os
import shutil
import argparse
import json
from datetime import datetime

def strip_directory_name(name):
    """
    サブディレクトリ名から末尾8文字を削除する。
    例:
      '20241229_181002_925622' -> '20241229_18100'
    """
    # nameの長さが8より短い場合はそのまま返す
    return name[:-8] if len(name) > 8 else name

def parse_timestamp(name):
    """
    サブディレクトリ名を datetime オブジェクトに変換（UTCマイクロ秒まで）
    例: '20241229_181002_925622'
    """
    try:
        return datetime.strptime(name, '%Y%m%d_%H%M%S_%f')
    except ValueError:
        return None

def parse_timestamp_from_filename(filename):
    """
    画像ファイル名からタイムスタンプを抽出。
    フォーマット: camera_X_YYYYMMDD_HHMMSS_microseconds.jpg
    """
    try:
        base_name = filename.split('.')[0]  # 拡張子を除去
        # フォーマット: "camera_X_YYYYMMDD_HHMMSS_microseconds"
        parts = base_name.split('_')
        print(f"分割結果: {parts}")  # デバッグ用
        if len(parts) >= 5:  # 分割後の要素数を確認
            camera = parts[0] + "_" + parts[1]  # "camera_X" 部分
            date = parts[2]  # YYYYMMDD
            time_micro = parts[3] + parts[4]  # HHMMSS_microseconds
            time = time_micro[:6]  # HHMMSS
            microseconds = time_micro[6:]  # microseconds
            print(f"解析中: camera={camera}, date={date}, time_micro={time_micro}")
            print(f"時間: {time}, マイクロ秒: {microseconds}")
            # フルタイムスタンプを生成
            full_datetime = f"{date}_{time}_{microseconds}"
            print(f"フルタイムスタンプ文字列: {full_datetime}")
            # datetime オブジェクトに変換
            timestamp = datetime.strptime(full_datetime, "%Y%m%d_%H%M%S_%f")
            return timestamp
        else:
            print(f"ファイル名が想定形式と一致しません: {filename}")
            return None
    except (ValueError, IndexError) as e:
        print(f"タイムスタンプの解析に失敗: {filename} -> エラー: {e}")
        return None





def calculate_offsets_with_multiple_cameras(image_dir, start_time):
    """
    複数のカメラで対応するタイムスタンプの平均を利用してオフセットを計算。
    """
    camera_timestamps = {}

    # 各カメラのタイムスタンプを収集
    for root, _, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith('.jpg'):
                image_timestamp = parse_timestamp_from_filename(filename)
                if image_timestamp:
                    camera_key = filename.split('_')[0] + '_' + filename.split('_')[1]  # 例: camera_0
                    if camera_key not in camera_timestamps:
                        camera_timestamps[camera_key] = []
                    camera_timestamps[camera_key].append(image_timestamp)

    # 全てのカメラのタイムスタンプをソート
    for key in camera_timestamps:
        camera_timestamps[key].sort()

    # 各カメラのタイムスタンプを平均化
    offsets = []
    min_length = min(len(timestamps) for timestamps in camera_timestamps.values())
    for i in range(min_length):
        avg_timestamp_seconds = sum(
            camera_timestamps[key][i].timestamp() for key in camera_timestamps
        ) / len(camera_timestamps)
        avg_timestamp = datetime.fromtimestamp(avg_timestamp_seconds)
        offset = (avg_timestamp - start_time).total_seconds() * 1_000_000  # マイクロ秒単位
        offsets.append(str(int(offset)))

    return offsets




def find_and_process_matching_directories(base_directory, output_directory):
    events_dir = os.path.join(base_directory, 'events')
    images_dir = os.path.join(base_directory, 'images')

    if not os.path.exists(events_dir) or not os.path.exists(images_dir):
        print("指定されたディレクトリに 'events' または 'images' が存在しません。")
        return

    events_subdirs = {strip_directory_name(name): name for name in os.listdir(events_dir)}
    images_subdirs = {strip_directory_name(name): name for name in os.listdir(images_dir)}

    for stripped_name in events_subdirs.keys() & images_subdirs.keys():
        events_full_name = events_subdirs[stripped_name]
        images_full_name = images_subdirs[stripped_name]

        events_path = os.path.join(events_dir, events_full_name)
        images_path = os.path.join(images_dir, images_full_name)

        event_start_time = parse_timestamp(events_full_name)
        rgb_image_start_time = parse_timestamp(images_full_name)

        target_dir = os.path.join(output_directory, events_full_name)
        events_target = os.path.join(target_dir, 'events')
        images_target = os.path.join(target_dir, 'images')

        os.makedirs(events_target, exist_ok=True)
        os.makedirs(images_target, exist_ok=True)

        for file_name in os.listdir(events_path):
            shutil.move(os.path.join(events_path, file_name), events_target)

        for file_name in os.listdir(images_path):
            shutil.move(os.path.join(images_path, file_name), images_target)

        offsets = calculate_offsets_with_multiple_cameras(images_target, rgb_image_start_time)

        offsets_file = os.path.join(target_dir, "image_offsets.txt")
        with open(offsets_file, "w", encoding="utf-8") as file:
            for offset in offsets:
                file.write(offset + "\n")

        metadata = {
            "events_path": events_target,
            "images_path": images_target,
            "event_start_time": event_start_time.isoformat() + "Z" if event_start_time else None,
            "rgb_image_start_time": rgb_image_start_time.isoformat() + "Z" if rgb_image_start_time else None,
            "offsets_file": offsets_file
        }
        metadata_file = os.path.join(target_dir, "metadata.json")
        with open(metadata_file, "w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=4, ensure_ascii=False)

        os.rmdir(events_path)
        os.rmdir(images_path)



def main():
    parser = argparse.ArgumentParser(
        description="Find, move, and process matching directory pairs in 'events' and 'images'."
    )
    parser.add_argument(
        "-i", "--input_directory", 
        required=True, 
        help="Base directory containing 'events' and 'images'."
    )
    parser.add_argument(
        "-o", "--output_directory", 
        required=True, 
        help="Output directory to move and process matching pairs."
    )
    
    args = parser.parse_args()
    base_directory = args.input_directory
    output_directory = args.output_directory
    
    find_and_process_matching_directories(base_directory, output_directory)

if __name__ == "__main__":
    main()
