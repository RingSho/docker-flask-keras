FROM tensorflow/tensorflow:1.13.1-gpu-py3
ARG project_dir=/projects/

ADD requirements_gpu.txt $project_dir
 
WORKDIR $project_dir

RUN apt-get update && apt-get install -y libsm6 libxrender1 libxext-dev
RUN pip install --upgrade pip && pip install -r requirements_gpu.txt