 #!/bin/bash

# --------------------------------------------------
# example script using nicMSlesions on Linux Systems
#
# Sergi Valverde 2018
# --------------------------------------------------

# check if the docker is running
DOCKER_NAME='nicmslesions'
DOCKER_RUNNING=`docker images | grep nicmslesions | wc -w`

if [ $DOCKER_RUNNING -gt 0 ];
then
    nvidia-docker run -ti  \
                  -v config:/home/user/config \
                  -v /:/data:rw \
                  nicmslesions python -u nic_train_network_batch.py --docker | tee log.txt
else
    docker build -f Dockerfile -t nicmslesions . &&
    nvidia-docker run -ti  \
                  -v config:/home/user/config \
                  -v /:/data:rw \
                  nicmslesions python -u nic_train_network_batch.py --docker | tee log.txt
fi
