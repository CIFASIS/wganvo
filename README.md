WGANVO
=============================

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


# Installation
1. Clone the repository
2. Run:
```
cd wganvo
sudo make image
sudo make start
```


<!--Para correr el test `vgg_trainable/test/test_model.py`, guardar las imágenes y el modelo en `images_dir` buscar donde se creo el volume, y en el shell del Docker, correr el test apuntando al volume. --> 



