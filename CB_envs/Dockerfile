FROM ros:noetic

LABEL org.opencontainers.image.authors="ohin"

ARG TZ=Asia/Taipei

ENV SHELL /bin/bash

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    git \
    sudo \
    lsof \
    cmake \
    python3-pip \
    v4l-utils \
    apt-transport-https \
    ros-noetic-tf* \
    ros-noetic-rviz \
    ros-noetic-diagnostic-updater \
    libgl1-mesa-glx libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev \
    libqt5gui5 libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get clean all

RUN pip3 install numpy \
    && pip3 install opencv-python \
    && pip3 install pyrealsense2

#install pytorch and ultralytics
RUN pip3 install --upgrade pip python-dateutil \
    && pip3 install pillow \
    && pip3 install torch torchvision \
    && pip3 install ultralytics

#Install Intel RealSense SDK 2.0(v2.50.0)
RUN git clone --branch v2.50.0 https://github.com/IntelRealSense/librealsense.git /opt/librealsense 
WORKDIR /opt/librealsense/build
RUN cmake ../ -DBUILD_EXAMPLES=true -DCMAKE_BUILD_TYPE=Release \
    && make -j4 && sudo make install

#clone ros pkgs
WORKDIR /root/realsense-ws/src
RUN git clone https://github.com/HungPinChiu/yolo && \
    apt update && \
    rosdep update && \
    rosdep install --from-paths . --ignore-src -r -y && \
    cd .. && \
    /bin/bash -c 'source /opt/ros/$ROS_DISTRO/setup.bash; catkin_make' && \
    echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc && \
    echo "source ~/realsense-ws/devel/setup.bash" >> ~/.bashrc && \
    rm -rf /var/lib/apt/lists/* 

COPY entrypoint.sh /entrypoint.sh
RUN  chmod +x /entrypoint.sh
ENTRYPOINT "/entrypoint.sh"

SHELL ["/bin/bash","-ic"]

