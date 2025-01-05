#!/bin/bash


# 10秒待機
sleep 10
# ROS 2サービスを用いて記録を開始する
echo "Starting recording..."
ros2 service call /set_recording std_srvs/srv/SetBool "{data: true}"

sleep 180
ros2 service call /set_recording std_srvs/srv/SetBool "{data: false}"


echo "Recording stopped."

