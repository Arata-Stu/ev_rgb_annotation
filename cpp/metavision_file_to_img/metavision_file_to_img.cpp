/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A.                                                                                       *
 *                                                                                                                    *
 * Licensed under the Apache License, Version 2.0 (the "License");                                                    *
 * you may not use this file except in compliance with the License.                                                   *
 * You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0                                 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License is distributed   *
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                      *
 * See the License for the specific language governing permissions and limitations under the License.                 *
 **********************************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <regex>
#include <thread>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/facilities/i_event_frame_decoder.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/raw_event_frame_converter.h>

namespace po = boost::program_options;

void save_frame_as_image(const cv::Mat &frame, const std::string &output_dir, int frame_number) {
    std::ostringstream filename;
    filename << output_dir << "/frame_" << std::setw(6) << std::setfill('0') << frame_number << ".jpg";
    if (!cv::imwrite(filename.str(), frame)) {
        MV_LOG_ERROR() << "Failed to save frame as image: " << filename.str();
    }
}

int main(int argc, char *argv[]) {
    std::string in_event_file_path;
    std::string output_image_dir;

    uint32_t accumulation_time;
    double slow_motion_factor;
    const std::uint16_t fps(30);

    const std::string program_desc(
        "Tool to generate images from a RAW or HDF5 file.\n\n"
        "The frame rate of the output is fixed to 30 FPS.\n"
        "Use the slow motion factor (-s option) to adjust the playback speed.\n"
        "Frames will be saved as JPEG images in the specified output directory.\n");

    po::options_description options_desc("Options");
    // clang-format off
    options_desc.add_options()
        ("help,h", "Produce help message.")
        ("input-event-file,i",   po::value<std::string>(&in_event_file_path)->required(), "Path to input event file (RAW or HDF5).")
        ("output-image-dir,o",   po::value<std::string>(&output_image_dir)->required(), "Directory to save output images. Frames will be saved as JPEG files.")
        ("accumulation-time,a",  po::value<uint32_t>(&accumulation_time)->default_value(10000), "Accumulation time (in us).")
        ("slow-motion-factor,s", po::value<double>(&slow_motion_factor)->default_value(1.), "Slow motion factor to apply to generate the frames.")
    ;
    // clang-format on

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(options_desc).run(), vm);
    if (vm.count("help")) {
        MV_LOG_INFO() << program_desc;
        MV_LOG_INFO() << options_desc;
        return 0;
    }

    try {
        po::notify(vm);
    } catch (po::error &e) {
        MV_LOG_ERROR() << program_desc;
        MV_LOG_ERROR() << options_desc;
        MV_LOG_ERROR() << "Parsing error:" << e.what();
        return 1;
    }

    if (slow_motion_factor <= 0) {
        MV_LOG_ERROR() << "Input slow motion factor must be greater than 0. Got" << slow_motion_factor;
        return 1;
    }

    Metavision::Camera camera;

    try {
        camera = Metavision::Camera::from_file(in_event_file_path, Metavision::FileConfigHints().real_time_playback(false));
    } catch (Metavision::CameraException &e) {
        MV_LOG_ERROR() << e.what();
        return 1;
    }

    // Get the geometry of the camera
    auto &geometry = camera.geometry();

    // Ensure the output directory exists
    if (!boost::filesystem::exists(output_image_dir)) {
        if (!boost::filesystem::create_directories(output_image_dir)) {
            MV_LOG_ERROR() << "Failed to create output directory: " << output_image_dir;
            return 1;
        }
    }

    Metavision::PeriodicFrameGenerationAlgorithm frame_generation(geometry.width(), geometry.height());
    bool has_cd = false;
    int frame_count = 0;

    try {
        auto &cd = camera.cd();
        has_cd = true;
        frame_generation.set_accumulation_time_us(accumulation_time);
        frame_generation.set_fps(slow_motion_factor * fps);
        frame_generation.set_output_callback(
            [&output_image_dir, &frame_count](Metavision::timestamp frame_ts, cv::Mat &cd_frame) {
                save_frame_as_image(cd_frame, output_image_dir, frame_count++);
            });

        cd.add_callback([&](const Metavision::EventCD *ev_begin, const Metavision::EventCD *ev_end) {
            frame_generation.process_events(ev_begin, ev_end);
        });
    } catch (...) {}

    camera.start();

    // Display a follow up message
    auto log = MV_LOG_INFO() << Metavision::Log::no_space << Metavision::Log::no_endline;
    const std::string message("Generating images...");
    int dots = 0;

    while (camera.is_running()) {
        log << "\r" << message.substr(0, message.size() - 3 + dots) + std::string("   ").substr(0, 3 - dots)
            << std::flush;
        dots = (dots + 1) % 4;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    log << "\rImages have been saved in " << output_image_dir << std::endl;
    return 0;
}
