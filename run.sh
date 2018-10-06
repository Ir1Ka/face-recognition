#! /usr/bin/env bash
DIR=$(cd $(dirname $0); pwd)
container_name=$(basename $DIR)
if which nvidia-docker 2>&1 1>/dev/null; then
    DOCKER_CMD=nvidia-docker
elif which docker 2>&1 1>/dev/null; then
    echo -e "\e[34;1mInfo\e[0m: Missing nvidia-docker, it will run in the TensorFlow of the CPU variant."
    echo -e "\tIf you have an NVIDIA graphics card installed on your computer, this may slow down."
    DOCKER_CMD=docker
else
    echo -e "\e[31;1mError\e[0m: Please install docker and nvidia-docker(2) if you have installed a NVIDIA graphics card."
    exit 1
fi
#DOCKER_OPTIONS='--privileged=true'
TF_IMAGE=`docker image ls tensorflow/tensorflow -q | head -n 1`
if [ -z "$TF_IMAGE" ]; then
    echo -e "\e[31;1mError\e[0m: Please install a tensorflow/tensorflow docker image for this project."
    echo -e "\tYou can run follow command to pull it."
    echo -e "\t\e[1mdocker pull tensorflow/tensorflow:latest-py3\e[0m"
    exit 1
fi
exec $DOCKER_CMD run $DOCKER_OPTIONS --name $container_name --rm -it -p 8888:8888 -p 6006:6006 -v $DIR:/notebooks $TF_IMAGE
