FROM tensorflow/tensorflow:1.8.0-devel-gpu
WORKDIR /app
COPY . /app
RUN ln -s /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so \
          /usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/libcuda.so.1 && \
    apt-get update && \
    apt-get install -y python-tk && \
    pip install --trusted-host pypi.python.org -r requirements.txt

ENV LD_LIBRARY_PATH="/usr/local/cuda-9.0/targets/x86_64-linux/lib/stubs/:${LD_LIBRARY_PATH}"
#RUN mkdir -p /var/kitti-images/
#VOLUME /var/kitti-images/
#CMD ["/bin/bash"]
