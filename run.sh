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

# For get image name.
format="--format {{.Repository}}:{{.Tag}}"

DOCKER_OPTIONS=("run")
#DOCKER_OPTIONS+=('--privileged=true')
DOCKER_OPTIONS+=("--name $container_name")
DOCKER_OPTIONS+=("--rm")
DOCKER_OPTIONS+=("-it")
DOCKER_OPTIONS+=("-p 8888:8888")
DOCKER_OPTIONS+=("-p 6006:6006")
DOCKER_OPTIONS+=("-v $DIR:/notebooks")

while getopts "i:o:" arg
do
    case "$arg" in
        i) # Image name
            TF_IMAGE=$($DOCKER_CMD image ls $OPTARG $format | head -n 1)
            if [ -z "$TF_IMAGE" ]; then
                echo -e "\e[31;1mWarning\e[0m: The image you specified does not exist, use the default image."
            fi
            ;;
        o) # Docker options.
            DOCKER_OPTIONS+=("$OPTARG")
            ;;
        *)
            ;;
    esac
done

if [ -z "$TF_IMAGE" ]; then
    TF_IMAGE=$($DOCKER_CMD image ls tensorflow/tensorflow $format | head -n 1)
fi

if [ -z "$TF_IMAGE" ]; then
    echo -e "\e[31;1mError\e[0m: Please install a tensorflow/tensorflow docker image for this project."
    echo -e "\tYou can run follow command to pull it."
    echo -e "\t\e[1mdocker pull tensorflow/tensorflow:latest-py3\e[0m"
    exit 1
fi

DOCKER_OPTIONS+=("$TF_IMAGE")

exec $DOCKER_CMD ${DOCKER_OPTIONS[@]}
