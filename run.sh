#! /usr/bin/env bash
container_name=$(basename $(pwd))
docker run --runtime="nvidia" --name $container_name --rm -it -p 8888:8888 -p 6006:6006 -v `pwd`:/notebooks tensorflow/tensorflow:latest-gpu-py3
#docker run --runtime="nvidia" --privileged=true --name $container_name --rm -it -p 8888:8888 -p 6006:6006 -v `pwd`:/notebooks tensorflow/tensorflow:latest-gpu-py3
