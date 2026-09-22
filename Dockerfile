FROM continuumio/miniconda3
WORKDIR /home/bomberman
RUN apt-get update
RUN apt-get -y install gcc g++
# The floating base can move beyond TensorFlow's supported Python versions.
# Avoid default-channel ToS prompts in unattended builds; inference uses CPU.
RUN conda create --prefix /opt/bomberman --override-channels -c conda-forge python=3.12 scipy numpy matplotlib numba
ENV PATH=/opt/bomberman/bin:$PATH
RUN conda install --prefix /opt/bomberman --override-channels -c conda-forge pytorch-cpu torchvision
RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm
RUN pip install pathfinding pyaml igraph ujson
RUN conda install --prefix /opt/bomberman --override-channels -c conda-forge pandas
RUN pip install networkx dill pyastar2d easydict sympy pygame
COPY . .
CMD /bin/bash
