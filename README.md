WGANVO: Monocular Visual Odometry based on Generative Adversarial Networks
=============================
Visual Odometry is one the most essential techniques for robot localization.
In this work we propose the use of Generative Adversarial Networks to estimate the pose taking images of a monocular camera as input. We present WGANVO, a Deep Learning based monocular Visual Odometry method. In particular, a neural network is trained to regress a pose estimate from an image pair. The training is performed using a semi-supervised approach, combining the unsupervised GAN technique with labeled data. Unlike geometry based monocular methods, the proposed method can recover the absolute scale of the observed scene without neither prior knowledge nor extra information as it can infer it from the training stage. The evaluation of the resulting system is carried out on the well-known KITTI dataset where it is shown to work in real time and the accuracy obtained is encouraging to continue the development of Deep Learning based methods.

### Demo
* https://www.youtube.com/watch?v=6vcR9PCsWDQ
* https://www.youtube.com/watch?v=zg5BlvUQhWE

# Requirements
* NVIDIA GPU
* Python 2.7 and pip (image pre-processing)

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

As a result, several files are generated. These files in this specific format are required to train the network. Future versions will no longer require this specific folder structure to be used.

## Training
Input must be provided in a specific folder structure. **This step will be optional in future versions.**

For example, if we want to train the network with sequences 00, 01 and 03 as input, we pre-process the images with ```adapt_images_kitti``` and then we save the output files in this way:
```
train_images/
├── 00
│   ├── images.npz
│   ├── t.npz
│   ├── images_shape.txt
│   └── ...
├── 01
│   ├── images.npz
│   ├── t.npz
│   ├── images_shape.txt
│   └── ...
└── 03
    ├── images.npz
    ├── t.npz
    ├── images_shape.txt
    └── ...

```
**Note**: Folder names are not required to be the same as the ones in this example.


<!--Para correr el test `vgg_trainable/test/test_model.py`, guardar las imágenes y el modelo en `images_dir` buscar donde se creo el volume, y en el shell del Docker, correr el test apuntando al volume. -->
