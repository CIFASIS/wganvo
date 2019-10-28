FROM tensorflow/tensorflow:1.8.0-devel-gpu
WORKDIR /app
COPY . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
#RUN mkdir -p /var/kitti-images/
#VOLUME /var/kitti-images/
#CMD ["/bin/bash"]
