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

RUN conda create --name railrl python=3.7.6 pip
RUN echo "source activate railrl" >> ~/.bashrc
# Use the railrl pip
ENV OLDPATH $PATH
ENV PATH /opt/conda/envs/railrl/bin:$PATH

RUN conda install -n railrl _libgcc_mutex=0.1=main
RUN conda install -n railrl _tflow_select=2.1.0=gpu
RUN conda install -n railrl absl-py=0.8.1=py37_0
RUN conda install -n railrl astor=0.8.0=py37_0
RUN conda install -n railrl astor=0.8.0=py37_0
RUN conda install -n railrl blas=1.0=mkl
RUN conda install -n railrl c-ares=1.15.0=h7b6447c_1001
RUN conda install -n railrl ca-certificates=2020.6.24=0
RUN conda install -n railrl certifi=2020.6.20=py37_0
RUN conda install -n railrl click=7.1.2=py_0
RUN conda install -n railrl cudatoolkit=10.0.130=0
RUN conda install -n railrl cudnn=7.6.5=cuda10.0_0
RUN conda install -n railrl cupti=10.0.130=0
RUN conda install -n railrl cycler=0.10.0=py37_0
RUN conda install -n railrl dbus=1.13.12=h746ee38_0
RUN conda install -n railrl expat=2.2.6=he6710b0_0
RUN conda install -n railrl fontconfig=2.13.0=h9420a91_0
RUN conda install -n railrl freetype=2.9.1=h8a8886c_1
RUN conda install -n railrl gast=0.2.2=py37_0
RUN conda install -n railrl glib=2.56.2=hd408876_0
RUN conda install -n railrl google-pasta=0.1.8=py_0
RUN conda install -n railrl grpcio=1.16.1=py37hf8bcb03_1
RUN conda install -n railrl gst-plugins-base=1.14.0=hbbd80ab_1
RUN conda install -n railrl gstreamer=1.14.0=hb453b48_1
RUN conda install -n railrl h5py=2.9.0=py37h7918eee_0
RUN conda install -n railrl hdf5=1.10.4=hb1b8bf9_0
RUN conda install -n railrl icu=58.2=h211956c_0
RUN conda install -n railrl intel-openmp=2019.4=243
RUN conda install -n railrl joblib=0.14.1=py_0
RUN conda install -n railrl jpeg=9b=h024ee3a_2
RUN conda install -n railrl keras-applications=1.0.8=py_0
RUN conda install -n railrl keras-preprocessing=1.1.0=py_1
RUN conda install -n railrl kiwisolver=1.1.0=py37he6710b0_0
RUN conda install -n railrl ld_impl_linux-64=2.33.1=h53a641e_7
RUN conda install -n railrl libedit=3.1.20181209=hc058e9b_0
RUN conda install -n railrl libffi=3.2.1=hd88cf55_4
RUN conda install -n railrl libgcc-ng=9.1.0=hdf63c60_0
RUN conda install -n railrl libgfortran-ng=7.3.0=hdf63c60_0
RUN conda install -n railrl libpng=1.6.37=hbc83047_0
RUN conda install -n railrl libprotobuf=3.11.2=hd408876_0
RUN conda install -n railrl libstdcxx-ng=9.1.0=hdf63c60_0
RUN conda install -n railrl libtiff=4.1.0=h2733197_0
RUN conda install -n railrl libuuid=1.0.3=h1bed415_2
RUN conda install -n railrl libxcb=1.13=h1bed415_1
RUN conda install -n railrl libxml2=2.9.9=hea5a465_1
RUN conda install -n railrl markdown=3.1.1=py37_0
RUN conda install -n railrl matplotlib=3.1.3=py37_0
RUN conda install -n railrl matplotlib-base=3.1.3=py37hef1b27d_0
RUN conda install -n railrl mkl=2019.4=243
RUN conda install -n railrl mkl-service=2.3.0=py37he904b0f_0
RUN conda install -n railrl mkl_fft=1.0.15=py37ha843d7b_0
RUN conda install -n railrl mkl_random=1.1.0=py37hd6b4f25_0
RUN conda install -n railrl ncurses=6.1=he6710b0_1
RUN conda install -n railrl ninja=1.9.0=py37hfd86e86_0
RUN conda install -n railrl numpy=1.18.1=py37h4f9e942_0
RUN conda install -n railrl numpy-base=1.18.1=py37hde5b4d6_0
RUN conda install -n railrl olefile=0.46=py_0
RUN conda install -n railrl openssl=1.1.1g=h7b6447c_0
RUN conda install -n railrl opt_einsum=3.1.0=py_0
RUN conda install -n railrl pandas=1.0.1=py37h0573a6f_0
#RUN conda install -n railrl patsy=0.5.1=py_0
RUN conda install -n railrl pcre=8.43=he6710b0_0
RUN conda install -n railrl pillow=7.0.0=py37hb39fc2d_0
RUN conda install -n railrl pip=19.3.1=py37_0
RUN conda install -n railrl protobuf=3.11.2=py37he6710b0_0
RUN conda install -n railrl pyparsing=2.4.6=py_0
RUN conda install -n railrl pyqt=5.9.2=py37h22d08a2_1
#RUN conda install -n railrl python=3.7.6=h0371630_2
RUN conda install -n railrl python-dateutil=2.8.1=py_0
#RUN conda install -n railrl python_abi=3.7=1_cp37m
RUN conda install -n railrl pytz=2019.3=py_0
RUN conda install -n railrl qt=5.9.7=h5867ecd_1
RUN conda install -n railrl readline=7.0=h7b6447c_5
RUN conda install -n railrl scikit-learn=0.22.1=py37hd81dba3_0
RUN conda install -n railrl scipy=1.3.2=py37h7c811a0_0
RUN conda install -n railrl seaborn=0.10.0=py_0
RUN conda install -n railrl setuptools=44.0.0=py37_0
#RUN conda install -n railrl sip=4.19.13=py37he6710b0_0
RUN conda install -n railrl six=1.13.0=py37_0
RUN conda install -n railrl sqlite=3.30.1=h7b6447c_0
#RUN conda install -n railrl statsmodels=0.11.1=py37h8f50634_2
RUN conda install -n railrl tensorflow-base=2.0.0=gpu_py37h0ec5d1f_0
RUN conda install -n railrl tensorflow-gpu=2.0.0=h0d30ee6_0
RUN conda install -n railrl termcolor=1.1.0=py37_1
RUN conda install -n railrl tk=8.6.8=hbc83047_0
RUN conda install -n railrl tornado=6.0.3=py37h7b6447c_3
RUN conda install -n railrl werkzeug=0.16.0=py_0
RUN conda install -n railrl wheel=0.33.6=py37_0
RUN conda install -n railrl wrapt=1.11.2=py37h7b6447c_0
RUN conda install -n railrl xz=5.2.4=h14c3975_4
RUN conda install -n railrl zlib=1.2.11=h7b6447c_3
RUN conda install -n railrl zstd=1.3.7=h0b5b093_0

RUN pip install alabaster==0.7.12
RUN pip install atari-py==0.2.6
RUN pip install attrs==19.3.0
RUN pip install babel==2.8.0
RUN pip install backcall==0.1.0
#RUN pip install baselines==0.1.5
RUN pip install cffi==1.14.0
RUN pip install chardet==3.0.4
RUN pip install cloudpickle==1.2.2
RUN pip install cython==0.29.19
RUN pip install decorator==4.4.2
RUN pip install dill==0.3.2
RUN pip install docutils==0.16
RUN pip install fasteners==0.15
RUN pip install future==0.18.2
RUN pip install glfw==1.11.1
RUN pip install gputil==1.4.0
RUN pip install gtimer==1.0.0b5
RUN pip install gym==0.15.4
RUN pip install idna==2.9
RUN pip install imagehash==4.1.0
RUN pip install imageio==2.8.0
RUN pip install imagesize==1.2.0
RUN pip install importlib-metadata==1.6.0
RUN pip install ipdb==0.13.2
RUN pip install ipython==7.14.0
RUN pip install ipython-genutils==0.2.0
RUN pip install jedi==0.17.0
RUN pip install jinja2==2.11.2
RUN pip install lockfile==0.12.2
RUN pip install lz4==3.0.2
RUN pip install markupsafe==1.1.1
RUN pip install monotonic==1.5
RUN pip install more-itertools==8.3.0
RUN pip install mpi4py==3.0.3
RUN pip install mujoco-py==2.0.2.10
RUN pip install numpydoc==1.0.0
RUN pip install opencv-python==4.1.2.30
RUN pip install opt-einsum==3.2.1
RUN pip install packaging==20.4
RUN pip install parso==0.7.0
RUN pip install pexpect==4.8.0
RUN pip install pickleshare==0.7.5
RUN pip install pluggy==0.13.1
RUN pip install progressbar2==3.51.4
RUN pip install prompt-toolkit==3.0.5
RUN pip install ptyprocess==0.6.0
RUN pip install py==1.8.1
RUN pip install pycparser==2.20
RUN pip install pyglet==1.3.2
RUN pip install pygments==2.6.1
RUN pip install pytest==5.4.2
RUN pip install pytest-instafail==0.3.0
RUN pip install python-utils==2.4.0
RUN pip install pywavelets==1.1.1
RUN pip install pyzmq==19.0.2
RUN pip install requests==2.23.0
RUN pip install setproctitle==1.1.10
RUN pip install snowballstemmer==2.0.0
RUN pip install sphinx==3.0.4
RUN pip install sphinx-rtd-theme==0.4.3
RUN pip install sphinxcontrib-applehelp==1.0.2
RUN pip install sphinxcontrib-devhelp==1.0.2
RUN pip install sphinxcontrib-htmlhelp==1.0.3
RUN pip install sphinxcontrib-jsmath==1.0.1
RUN pip install sphinxcontrib-qthelp==1.0.3
RUN pip install sphinxcontrib-serializinghtml==1.1.4
RUN pip install tensorboard==1.15.0
RUN pip install tensorboardx==1.8
RUN pip install tensorflow==1.15.2
RUN pip install tensorflow-estimator==1.15.1
RUN pip install torch==1.3.0
RUN pip install torchvision==0.4.1
RUN pip install tqdm==4.48.2
RUN pip install traitlets==4.3.3
RUN pip install urllib3==1.25.9
RUN pip install wcwidth==0.1.9
RUN pip install zipp==3.1.0
RUN pip install zmq==0.0.0

RUN pip install cloudpickle==0.5.2


########
### Mujoco-py and gym
########
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
RUN pip install gym[all]==0.12.5

##########################################################
### Cleanup
##########################################################

# Technically this shouldn't be necessary since we put this in the bashrc
# but for some reason, bashrc isn't being run on AWS.
ENTRYPOINT source activate railrl
