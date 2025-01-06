#!/bin/bash

# 引数が指定されているかチェック
if [ -z "$1" ]; then
  echo "使用方法: $0 <input_dir> <output_dir> <homography_matrix.yaml>"
  exit 1
fi

# 引数から入力ディレクトリを取得
INPUT_DIR="$1"
OUTPUT_DIR="$2"
HOMOGRAPHY_MATRIX="$3"

# ディレクトリが存在するかチェック
if [ ! -d "$INPUT_DIR" ]; then
  echo "エラー: 指定したディレクトリが存在しません。"
  exit 1
fi

# 出力ディレクトリが存在するかチェック
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "エラー: 指定したディレクトリが存在しません。"
  exit 1
fi

# ファイルが存在するかチェック
if [ ! -f "$HOMOGRAPHY_MATRIX" ]; then
  echo "エラー: 指定したファイルが存在しません。"
  exit 1
fi

# 仮想環境がアクティブかチェック
if [ -n "$VIRTUAL_ENV" ]; then
    echo "仮想環境がアクティブです: $VIRTUAL_ENV"
else
    echo "仮想環境をアクティブにします"
    source env/bin/activate
fi

# python scriptを実行
## データセット　ディレクトリを移動
python3 python/match_dataset.py -i "$INPUT_DIR" -o "$OUTPUT_DIR"

# 環境変数をexport
export HDF5_PLUGIN_PATH=$HDF5_PLUGIN_PATH:/usr/lib/x86_64-linux-gnu/hdf5/serial/plugins

## raw ファイルをhdf5に変換
metavision_file_to_hdf5 -i "$OUTPUT_DIR" -r -p ".*\\.raw" 



for dir in "$OUTPUT_DIR"/*/; do
  if [ -d "$dir" ]; then
    # ダブルスラッシュを防ぐため、パスを正規化
    normalized_dir=$(realpath "$dir")
    echo "$normalized_dir"

    ## フレームをアノテーション
    python3 python/track.py -b "$normalized_dir" 

    ## アノテーションを変換
    python3 python/convert_labels.py -b "$normalized_dir" -m "$HOMOGRAPHY_MATRIX"


  fi

## 終了
echo "処理が完了しました。"
done
