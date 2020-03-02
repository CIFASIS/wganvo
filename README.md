WGANVO
=============================

# Requirements
NVIDIA GPU
Python 2.7 and pip (image pre-processing)

# Dependencies 
## Docker and docker-compose
1. Install Docker and docker-compose

2. Install nvidia-docker:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Dependencies for pre-processing KITTI images
Run:
```
pip install -r requirements.txt
```


# Installation
1. Clone the repository
2. Run:
```
cd wganvo
sudo make image
sudo make start
```

# KITTI
## Image pre-processing
In order to reduce the resolution of the images, we pre-process KITTI images using a Python script. 
**This step will be optional in future versions.**

1. Download KITTI odometry dataset
2. For each of the KITTI sequences, simply run:
```
python adapt_images_kitti <path-to-sequence-dir> <path-to-poses-file> --crop 500 375 --scale 128 96 --output_dir <path-to-output-dir> 
```



<!--Para correr el test `vgg_trainable/test/test_model.py`, guardar las imágenes y el modelo en `images_dir` buscar donde se creo el volume, y en el shell del Docker, correr el test apuntando al volume. --> 



