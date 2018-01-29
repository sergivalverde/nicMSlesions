 #!/bin/bash

# --------------------------------------------------
# example script using nicMSlesions on Linux Systems
#
# Sergi Valverde 2018
# --------------------------------------------------


# check new versions
docker pull nicvicorob/mslesions:latest

nvidia-docker run -ti  \
              -v config:/home/user/config \
              -v /:/data:rw \
              nicvicorob/mslesions:latest python -u nic_train_network_batch.py --docker | tee log.txt
