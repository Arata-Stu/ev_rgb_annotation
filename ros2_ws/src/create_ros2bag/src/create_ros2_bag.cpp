#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/opencv.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_cpp/writers/sequential_writer.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <filesystem>
#include <vector>
#include <tuple>
#include <algorithm>
#include <rcutils/types/uint8_array.h>

namespace fs = std::filesystem;

// トピックを追加する関数
void add_topic(std::shared_ptr<rosbag2_cpp::writers::SequentialWriter> writer,
               const std::string &topic_name,
               const std::string &message_type) {
    rosbag2_storage::TopicMetadata topic_metadata{
        topic_name, message_type, "cdr", "sensor_msgs::msg::Image"};
    writer->create_topic(topic_metadata);
}

// PNGファイルからタイムスタンプを抽出
std::vector<std::tuple<std::string, int64_t, int64_t>> extract_timestamps_from_png(const std::vector<std::string> &file_paths) {
    std::vector<std::tuple<std::string, int64_t, int64_t>> timestamps;
    for (const auto &file_path : file_paths) {
        std::string basename = fs::path(file_path).stem().string();
        int64_t seconds = std::stoll(basename.substr(0, basename.size() - 9));
        int64_t nanoseconds = std::stoll(basename.substr(basename.size() - 9));
        timestamps.emplace_back(file_path, seconds, nanoseconds);
    }
    return timestamps;
}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("rosbag2_writer");

    // パラメータの宣言
    node->declare_parameter<std::string>("rosbag_folder", "/path/to/rosbag");
    node->declare_parameter<std::string>("mono_folder", "/path/to/mono");
    node->declare_parameter<std::string>("rgb_folder", "/path/to/rgb");
    node->declare_parameter<std::string>("rgb_folder_2", "");
    node->declare_parameter<std::string>("mono_topic", "/camera/mono/image_raw");
    node->declare_parameter<std::string>("rgb_topic", "/camera/rgb/image_raw");
    node->declare_parameter<std::string>("rgb_topic_2", "/camera/rgb2/image_raw");
    node->declare_parameter<int>("downsample_rate", 12);

    // パラメータの取得
    std::string rosbag_folder, mono_folder, rgb_folder, rgb_folder_2, mono_topic, rgb_topic, rgb_topic_2;
    int downsample_rate;
    node->get_parameter("rosbag_folder", rosbag_folder);
    node->get_parameter("mono_folder", mono_folder);
    node->get_parameter("rgb_folder", rgb_folder);
    node->get_parameter("rgb_folder_2", rgb_folder_2);
    node->get_parameter("mono_topic", mono_topic);
    node->get_parameter("rgb_topic", rgb_topic);
    node->get_parameter("rgb_topic_2", rgb_topic_2);
    node->get_parameter("downsample_rate", downsample_rate);

    // ロスバッグ作成
    auto writer = std::make_shared<rosbag2_cpp::writers::SequentialWriter>();
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = rosbag_folder;
    storage_options.storage_id = "sqlite3";

    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";

    writer->open(storage_options, converter_options);
    add_topic(writer, mono_topic, "sensor_msgs/msg/Image");
    add_topic(writer, rgb_topic, "sensor_msgs/msg/Image");
    if (!rgb_folder_2.empty()) {
        add_topic(writer, rgb_topic_2, "sensor_msgs/msg/Image");
    }

    // MonoとRGBの画像ファイルリストを取得
    std::vector<std::string> mono_files, rgb_files, rgb_files_2;
    for (const auto &entry : fs::directory_iterator(mono_folder)) {
        mono_files.push_back(entry.path().string());
    }
    for (const auto &entry : fs::directory_iterator(rgb_folder)) {
        rgb_files.push_back(entry.path().string());
    }
    if (!rgb_folder_2.empty()) {
        for (const auto &entry : fs::directory_iterator(rgb_folder_2)) {
            rgb_files_2.push_back(entry.path().string());
        }
    }

    std::sort(mono_files.begin(), mono_files.end());
    std::sort(rgb_files.begin(), rgb_files.end());
    if (!rgb_files_2.empty()) {
        std::sort(rgb_files_2.begin(), rgb_files_2.end());
    }

    // ダウンサンプリング
    std::vector<std::tuple<std::string, int64_t, int64_t>> mono_data = extract_timestamps_from_png(mono_files);
    mono_data.resize(mono_data.size() / downsample_rate);

    // イメージ処理と書き込み
    // RGB画像処理と書き込み
    for (size_t i = 0; i < mono_data.size(); ++i) {
        const auto &[mono_path, seconds, nanoseconds] = mono_data[i];
        // モノクロ画像処理（既存のコード）
        cv::Mat mono_img = cv::imread(mono_path, cv::IMREAD_GRAYSCALE);
        if (!mono_img.empty()) {
            auto mono_msg = std::make_shared<sensor_msgs::msg::Image>();
            mono_msg->header.stamp.sec = seconds;
            mono_msg->header.stamp.nanosec = nanoseconds;
            mono_msg->height = mono_img.rows;
            mono_msg->width = mono_img.cols;
            mono_msg->encoding = "mono8";
            mono_msg->step = mono_img.step;
            mono_msg->data.assign(mono_img.data, mono_img.data + (mono_img.rows * mono_img.cols));

            rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
            rclcpp::SerializedMessage serialized_data;
            serializer.serialize_message(mono_msg.get(), &serialized_data);

            auto serialized_msg = std::make_shared<rosbag2_storage::SerializedBagMessage>();
            serialized_msg->time_stamp = rclcpp::Time(seconds, nanoseconds, RCL_ROS_TIME).nanoseconds();
            serialized_msg->topic_name = mono_topic;
            serialized_msg->serialized_data = std::make_shared<rcutils_uint8_array_t>();
            serialized_msg->serialized_data->allocator = rcutils_get_default_allocator();
            serialized_msg->serialized_data->buffer_length = serialized_data.size();
            serialized_msg->serialized_data->buffer_capacity = serialized_data.capacity();
            serialized_msg->serialized_data->buffer = static_cast<uint8_t *>(malloc(serialized_data.size()));
            memcpy(serialized_msg->serialized_data->buffer, serialized_data.get_rcl_serialized_message().buffer, serialized_data.size());
            writer->write(serialized_msg);
        }

        // RGB画像処理 (rgb_folder)
        if (i < rgb_files.size()) {
            cv::Mat rgb_img = cv::imread(rgb_files[i], cv::IMREAD_COLOR);
            if (!rgb_img.empty()) {
                auto rgb_msg = std::make_shared<sensor_msgs::msg::Image>();
                rgb_msg->header.stamp.sec = seconds;
                rgb_msg->header.stamp.nanosec = nanoseconds + 1;  // nanosec + 1
                rgb_msg->height = rgb_img.rows;
                rgb_msg->width = rgb_img.cols;
                rgb_msg->encoding = "rgb8";
                rgb_msg->step = rgb_img.step;
                rgb_msg->data.assign(rgb_img.data, rgb_img.data + (rgb_img.rows * rgb_img.step));

                rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
                rclcpp::SerializedMessage serialized_data;
                serializer.serialize_message(rgb_msg.get(), &serialized_data);

                auto serialized_msg = std::make_shared<rosbag2_storage::SerializedBagMessage>();
                serialized_msg->time_stamp = rclcpp::Time(seconds, nanoseconds + 1, RCL_ROS_TIME).nanoseconds();
                serialized_msg->topic_name = rgb_topic;
                serialized_msg->serialized_data = std::make_shared<rcutils_uint8_array_t>();
                serialized_msg->serialized_data->allocator = rcutils_get_default_allocator();
                serialized_msg->serialized_data->buffer_length = serialized_data.size();
                serialized_msg->serialized_data->buffer_capacity = serialized_data.capacity();
                serialized_msg->serialized_data->buffer = static_cast<uint8_t *>(malloc(serialized_data.size()));
                memcpy(serialized_msg->serialized_data->buffer, serialized_data.get_rcl_serialized_message().buffer, serialized_data.size());
                writer->write(serialized_msg);
            }
        }

        // RGB画像処理 (rgb_folder_2)
        if (!rgb_files_2.empty() && i < rgb_files_2.size()) {
            cv::Mat rgb_img_2 = cv::imread(rgb_files_2[i], cv::IMREAD_COLOR);
            if (!rgb_img_2.empty()) {
                auto rgb_msg_2 = std::make_shared<sensor_msgs::msg::Image>();
                rgb_msg_2->header.stamp.sec = seconds;
                rgb_msg_2->header.stamp.nanosec = nanoseconds + 2;  // nanosec + 2
                rgb_msg_2->height = rgb_img_2.rows;
                rgb_msg_2->width = rgb_img_2.cols;
                rgb_msg_2->encoding = "rgb8";
                rgb_msg_2->step = rgb_img_2.step;
                rgb_msg_2->data.assign(rgb_img_2.data, rgb_img_2.data + (rgb_img_2.rows * rgb_img_2.step));

                rclcpp::Serialization<sensor_msgs::msg::Image> serializer;
                rclcpp::SerializedMessage serialized_data;
                serializer.serialize_message(rgb_msg_2.get(), &serialized_data);

                auto serialized_msg = std::make_shared<rosbag2_storage::SerializedBagMessage>();
                serialized_msg->time_stamp = rclcpp::Time(seconds, nanoseconds + 2, RCL_ROS_TIME).nanoseconds();
                serialized_msg->topic_name = rgb_topic_2;
                serialized_msg->serialized_data = std::make_shared<rcutils_uint8_array_t>();
                serialized_msg->serialized_data->allocator = rcutils_get_default_allocator();
                serialized_msg->serialized_data->buffer_length = serialized_data.size();
                serialized_msg->serialized_data->buffer_capacity = serialized_data.capacity();
                serialized_msg->serialized_data->buffer = static_cast<uint8_t *>(malloc(serialized_data.size()));
                memcpy(serialized_msg->serialized_data->buffer, serialized_data.get_rcl_serialized_message().buffer, serialized_data.size());
                writer->write(serialized_msg);
            }
        }
    }

    rclcpp::shutdown();
    return 0;
}
