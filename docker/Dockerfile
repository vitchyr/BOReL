# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

SHELL ["/bin/bash", "-c"]

##########################################################
### System dependencies
##########################################################

# Now let's download python 3 and all the dependencies
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    curl \
    git \
    libav-tools \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget \
    xpra \
    xserver-xorg-dev \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Not sure why this is needed
ENV LANG C.UTF-8

# Not sure what this is fixing
COPY ./files/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./files/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Not sure why this is needed
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

##########################################################
### MuJoCo
##########################################################
# Note: ~ is an alias for /root
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && rm mujoco.zip
COPY ./files/mjkey.txt /root/.mujoco/mjkey.txt
RUN ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200
ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}



##########################################################
### Python
##########################################################
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN conda update -y --name base conda && conda clean --all -y


RUN conda create --name railrl python=3.7.4 pip
RUN echo "source activate railrl" >> ~/.bashrc
# Use the railrl pip
ENV OLDPATH $PATH
ENV PATH /opt/conda/envs/railrl/bin:$PATH


RUN pip install click==7.1.2
RUN pip install six==1.13.0
RUN pip install joblib==0.14.1
RUN pip install numpy==1.18.1
RUN pip install torch==1.10.0
RUN pip install tqdm==4.48.2
RUN pip install gym==0.12.5
RUN pip install matplotlib==3.1.3
RUN pip install requests==2.23.0
RUN pip install seaborn==0.10.0
RUN pip install tensorboardX==1.8
RUN pip install gitpython==2.1.7
RUN pip install mujoco-py==2.0.2.10
# RUN pip install pyglet==1.3.2
# RUN pip install gym==0.15.4


#RUN pip install absl-py==0.8.1
#RUN pip install alabaster==0.7.12
#RUN pip install astor==0.8.0
#RUN pip install attrs==19.3.0
#RUN pip install Babel==2.8.0
#RUN pip install backcall==0.1.0
#RUN pip install certifi==2020.6.20
#RUN pip install cffi==1.14.0
#RUN pip install chardet==3.0.4
#RUN pip install click==7.1.2
#RUN pip install cloudpickle==1.2.2
#RUN pip install cycler==0.10.0
#RUN pip install Cython==0.29.19
#RUN pip install decorator==4.4.2
#RUN pip install dill==0.3.2
#RUN pip install docutils==0.16
#RUN pip install fasteners==0.15
#RUN pip install future==0.18.2
#RUN pip install gast==0.2.2
#RUN pip install glfw==1.11.1
#RUN pip install google-pasta==0.1.8
#RUN pip install GPUtil==1.4.0
#RUN pip install grpcio==1.16.1
#RUN pip install gtimer==1.0.0b5
#RUN pip install gym==0.15.4
#RUN pip install h5py==2.9.0
#RUN pip install idna==2.9
#RUN pip install ImageHash==4.1.0
#RUN pip install imageio==2.8.0
#RUN pip install imagesize==1.2.0
#RUN pip install importlib-metadata==1.6.0
#RUN pip install ipdb==0.13.2
#RUN pip install ipython==7.14.0
#RUN pip install ipython-genutils==0.2.0
#RUN pip install jedi==0.17.0
#RUN pip install Jinja2==2.11.2
#RUN pip install joblib==0.14.1
#RUN pip install Keras-Applications==1.0.8
#RUN pip install Keras-Preprocessing==1.1.0
#RUN pip install kiwisolver==1.1.0
#RUN pip install lockfile==0.12.2
#RUN pip install lz4==3.0.2
#RUN pip install Markdown==3.1.1
#RUN pip install MarkupSafe==1.1.1
#RUN pip install matplotlib==3.1.3
##RUN pip install mkl-fft==1.0.15
##RUN pip install mkl-random==1.1.0
#RUN pip install mkl-service==2.3.0
#RUN pip install monotonic==1.5
#RUN pip install more-itertools==8.3.0
##RUN pip install mpi4py==3.0.3
#RUN pip install mujoco-py==2.0.2.10
#RUN pip install numpy==1.18.1
#RUN pip install numpydoc==1.0.0
#RUN pip install olefile==0.46
#RUN pip install opencv-python==4.1.2.30
#RUN pip install opt-einsum==3.2.1
#RUN pip install packaging==20.4
#RUN pip install pandas==1.0.1
#RUN pip install parso==0.7.0
#RUN pip install patsy==0.5.1
#RUN pip install pexpect==4.8.0
#RUN pip install pickleshare==0.7.5
#RUN pip install Pillow==7.0.0
#RUN pip install pluggy==0.13.1
#RUN pip install progressbar2==3.51.4
#RUN pip install prompt-toolkit==3.0.5
#RUN pip install protobuf==3.11.2
#RUN pip install ptyprocess==0.6.0
#RUN pip install py==1.8.1
#RUN pip install pycparser==2.20
#RUN pip install pyglet==1.3.2
#RUN pip install Pygments==2.6.1
#RUN pip install pyparsing==2.4.6
#RUN pip install pytest==5.4.2
#RUN pip install pytest-instafail==0.3.0
#RUN pip install python-dateutil==2.8.1
#RUN pip install python-utils==2.4.0
#RUN pip install pytz==2019.3
#RUN pip install PyWavelets==1.1.1
#RUN pip install pyzmq==19.0.2
#RUN pip install requests==2.23.0
#RUN pip install scikit-learn==0.22.1
#RUN pip install scipy==1.3.2
#RUN pip install seaborn==0.10.0
#RUN pip install setproctitle==1.1.10
##RUN pip install sip==4.19.13
#RUN pip install six==1.13.0
#RUN pip install snowballstemmer==2.0.0
#RUN pip install Sphinx==3.0.4
#RUN pip install statsmodels==0.11.1
#RUN pip install tensorboard==2.0.0
#RUN pip install tensorboardX==1.8
##RUN pip install tensorflow==2.0.0
##RUN pip install tensorflow-estimator==2.0.0
#RUN pip install termcolor==1.1.0
#RUN pip install torch==1.10.0
##RUN pip install torch==1.3.0
##RUN pip install torchvision==0.4.1
#RUN pip install tornado==6.0.3
#RUN pip install tqdm==4.48.2
#RUN pip install traitlets==4.3.3
#RUN pip install urllib3==1.25.9
#RUN pip install wcwidth==0.1.9
#RUN pip install Werkzeug==0.16.0
#RUN pip install wrapt==1.11.2
#RUN pip install zipp==3.1.0
#RUN pip install zmq==0.0.0
#WORKDIR /root
#COPY ./omrl.yml /root/omrl.yml
#RUN conda env create -f /root/omrl.yml
#RUN echo "source activate railrl" >> ~/.bashrc
#ENV OLDPATH $PATH
#ENV PATH /opt/conda/envs/railrl/bin:$PATH


########
### Mujoco-py and gym
########
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
#RUN pip install gym[all]==0.12.5
RUN python -c 'import mujoco_py'
