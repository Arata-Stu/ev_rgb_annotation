#

## setup

### 1. cpp build

```shell
cd cpp/events_to_img/
mkdir build && cd build
cmake ..
make
cd ../../..

cd cpp/metavision_file_to_img/
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### 2. ros2 build
```shell
sudo apt install python3-vcstool
vcs import ros2_ws/src < packages.repos

cd ./ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

### 3. python virtual env

```shell
python3.9 -m venv env
source env/bin/activate
pip3 install -r requirements.txt 
```

## calibration

### 1. recording
```shell
## terminal1
source ros2_ws/install/setup.bash
ros2 launch calibration_launch calibration.launch.xml 

## terminal2 toggle recording status
source ros2_ws/install/setup.bash
recording.sh
```

### 2. processing data
```shell
python3 match_dataset.py -i /home/arata22/recording/ -o /home/arata22/dataset/

metavision_file_to_hdf5 -i /home/arata22/dataset/  -r -p ".*\\.raw" 
```

### 3. reconstruct gray image from event
```shell
## clone and create conda env
git clone https://github.com/Arata-Stu/e2calib.git

cd e2calib
conda create -y -n e2calib python=3.7
conda activate e2calib
conda install -y -c anaconda numpy scipy
conda install -y -c conda-forge h5py opencv tqdm

pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html


## reconstruct
python offline_reconstruction.py  --upsample_rate 2 --h5file /path/to/hdf5 --output_folder /path/to/output_folder/ --timestamps_file /path/to/trigger.txt --height 480 --width 640
```

### 4. create rosbag
```shell
## create ros2 bag
ros2 launch create_ros2bag bag_creater.launch.xml

## convert ros2bag to ros1 bag
deactivate ## if activate virtual env
rosbags-convert ./path/to/combined_ros2/
```
### 5. calibrate using Kalibr
```shell
xhost +local:root
sudo docker run -it -e "DISPLAY" -e "QT_X11_NO_MITSHM=1" -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /path/to/rosbag:/calib stereolabs/kalibr:kinetic

kalibr_calibrate_cameras --bag /calib/combined_ros2.bag --target /calib/target.yaml --models 'pinhole-radtan' 'pinhole-radtan' --topics /event/image_raw /rgb_cam/image_raw
```