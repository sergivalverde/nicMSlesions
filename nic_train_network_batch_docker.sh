#!/bin/bash

# --------------------------------------------------
# example script using nicMSlesions on Linux Systems
#
# Sergi Valverde 2018
# --------------------------------------------------

# check if the docker is running
DOCKER_NAME='nicms_docker'
DOCKER_RUNNING=`docker images | grep nicmslesions | wc -w`

if [ $DOCKER_RUNNING -gt 0 ];
then  nvidia-docker run -ti  \
       -v config:/home/user/config \
       -v /:/data:rw \
       nicmslesions_user python -u nic_train_network_batch.py --docker
else
    docker build -f Dockerfile -t nicmslesions . &&
    nvidia-docker run -ti  \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v $DATAPATH:/data:rw \
       nicmslesions
fi
