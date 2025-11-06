# We get the right environment from docker repository
FROM eywalker/nvidia-cuda:8.0-cudnn5-devel-ubuntu16.04
#
#   This is a docker containter for building the original version of nicMSLesions and directly work
#   with the current configuration files of the main git repository from Sergi Valverde
#
#   To build it with different configuration files, fork the git repository, change your configuration.cfg file 
#   and adapt the line 49 accordingly
#
#   To build it, please, use the following command:
#      docker build --tag=nicmslesions-v1.0 -f `pwd`/DockerFile `pwd`
#
#      docker run --rm -it \
#		              -v `pwd`:/data/path/to/testing \
#                 nicmslesions-v1.0 \
#		              python2 nic_infer_segmentation_batch.py --docker 
#      
#   By default you should have all subject data in a directory, and one directory per subject.
#   In the same directory and the filenames should be consistent:
#   ms_people \ 
#              s1 \ 
#                 flair_s1.nii.gz t1_s1.nii.gz 
#              s2 \ 
#                 flair_s2.nii.gz t1_s2.nii.gz 
#   ... 
#              sn \ 
#                 flair_sN.nii.gz t1_sN.nii.gz 
#

# We setup the timezone
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
RUN apt-get update && apt-get install -y \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev \
  python-dev \
  python-tk\
  git \
  curl \
  emacs24

# Download nicMSLesions from the github repository and add it to the path
RUN git clone https://github.com/sergivalverde/nicMSlesions.git nicMSlesions
ENV PATH=/nicMSlesions:${PATH}

# Download and install miniconda to /miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh --no-check-certificate && \
    sh Miniconda3-4.7.12-Linux-x86_64.sh -p /miniconda -b && \
    rm /Miniconda3-4.7.12-Linux-x86_64.sh

# Add miniconda to the path environment variable
ENV PATH=/miniconda/bin:${PATH}

# We install an old pip but we don't upgrade it
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py 
RUN python2 get-pip.py
RUN python2.7 -m pip install --upgrade setuptools 

# We install the packages in this order for ensuring that we do not have version conflicts across dependencies
RUN python2.7 -m pip install h5py==2.10.0 
RUN python2.7 -m pip install numpy==1.16.6 
RUN python2.7 -m pip install scipy==1.0.0  
RUN python2.7 -m pip install scikit-learn==0.20.4  
RUN python2.7 -m pip install tensorflow==1.6.0  
RUN python2.7 -m pip install keras==2.0.0 
RUN python2.7 -m pip install medpy==0.3.0 
RUN python2.7 -m pip install nibabel==2.1.0 
RUN python2.7 -m pip install pillow==6.2.2 
RUN python2.7 -m pip install simpleitk==1.2.4 
RUN python2.7 -m pip install mkl

# Finally, we setup the default working dir and the data directory
RUN mkdir /data

# This is the default working directory when we run the docker
WORKDIR /nicMSlesions
