#!/bin/bash

# 環境変数の設定
DATA_DIR="/path/to/data"  # データセットのディレクトリ
DEST_DIR="/path/to/dest"  # 前処理後のデータ保存先
NUM_PROCESSES=1  # 並列処理数

# デバッグ用ログ出力
echo "DATA_DIR=${DATA_DIR}, DEST_DIR=${DEST_DIR}, NUM_PROCESSES=${NUM_PROCESSES}"

# エラーチェック
if [ ! -d "${DATA_DIR}" ]; then
  echo "Error: DATA_DIR does not exist -> ${DATA_DIR}"
  exit 1
fi

if [ ! -d "${DEST_DIR}" ]; then
  echo "Warning: DEST_DIR does not exist -> Creating ${DEST_DIR}"
  mkdir -p "${DEST_DIR}"
fi

# 実行コマンド
python preprocess_dataset.py "${DATA_DIR}" "${DEST_DIR}" \
  conf_preprocess/representation/event_frame.yaml \
  conf_preprocess/extraction/const_duration_20.yaml \
  conf_preprocess/filter_gifu.yaml \
  -ds gifu -np "${NUM_PROCESSES}"

# 実行結果のステータス確認
if [ $? -eq 0 ]; then
  echo "Preprocessing completed successfully!"
else
  echo "Error: Preprocessing failed!"
  exit 1
fi
