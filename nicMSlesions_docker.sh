#!/bin/bash

# --------------------------------------------------
# example script using nicMSlesions on Linux Systems
#
# Sergi Valverde 2018
# --------------------------------------------------

echo  "##################################################"
echo  "# ------------                                   #"
echo  "# nicMSlesions (docker edition)                  #"
echo  "# ------------                                   #"
echo  "# MS WM lesion segmentation                      #"
echo  "#                                                #"
echo  "# -------------------------------                #"
echo  "# (c) Sergi Valverde 2018                        #"
echo  "# Neuroimage Computing Group                     #"
echo  "# -------------------------------                #"
echo  "##################################################"
echo " "


if [ $# != 2 ];
then
    echo "Number of incorrect parameters."
    echo "Please, enter the data path: nicMSLesions_docker -d /path/to/data"
    exit
fi

if [ "$1" != "-d" ];
then
    echo "Please, enter the data path: nicMSLesions_docker -d /path/to/data"
else
    DATAPATH=$2
fi


# check new versions
docker pull nicvicorob/mslesions:latest

DOCKER_NAME='nicms_docker'
DOCKER_RUNNING=`docker images | grep nicmslesions | wc -w`
nvidia-docker run -ti  \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v $DATAPATH:/data:rw \
       nicvicorob/mslesions  python -u app.py --docker |  tee log.txt
