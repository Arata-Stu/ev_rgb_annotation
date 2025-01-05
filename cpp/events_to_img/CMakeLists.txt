cmake_minimum_required(VERSION 3.10)

# プロジェクト名とC++の標準を設定
project(events_to_img VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# HDF5ライブラリを探す
find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(OpenCV REQUIRED)

# ソースファイルを指定
set(SOURCES
    events_to_img.cpp # メインソースファイル
)

# 実行可能ファイルを作成
add_executable(${PROJECT_NAME} ${SOURCES})

# HDF5ライブラリをリンク
target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${HDF5_LIBRARIES} ${OpenCV_LIBS})

# コンパイル時の定義とオプション（必要に応じて調整）
target_compile_definitions(${PROJECT_NAME} PRIVATE ${HDF5_DEFINITIONS})
