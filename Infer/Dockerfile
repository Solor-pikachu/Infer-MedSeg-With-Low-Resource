# FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
# XXX: IN CASE apt-get update fail, you can uncomment below two lines
# RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list  && \
#    sed -i s@/security.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
# RUN echo [global]'\n'index-url = https://mirrors.aliyun.com/pypi/simple/ > /etc/pip.conf

# RUN apt-get update && apt-get install -y --no-install-recommends \
# 	# we have found python3.7 in base docker
# 	python3-pip \
# 	python3-setuptools \
# 	build-essential \
# 	&& \
# 	apt-get clean && \
# 	python -m pip install --upgrade pip

WORKDIR /workspace
ADD ./   /workspace
RUN pip config set global.index-url https://pypi.douban.com/simple/
# RUN pip install pip -U


RUN pip install scikit-image
RUN pip install SimpleITK
RUN pip install scipy
RUN pip install numpy
RUN pip install connected-components-3d
RUN pip install fastremap
RUN python load_env.py

CMD ["bash", "predict.sh"]