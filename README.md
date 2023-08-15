WGANVO: Monocular Visual Odometry based on Generative Adversarial Networks
=============================
Visual Odometry is one the most essential techniques for robot localization.
In this work we propose the use of Generative Adversarial Networks to estimate the pose taking images of a monocular camera as input. We present WGANVO, a Deep Learning based monocular Visual Odometry method. In particular, a neural network is trained to regress a pose estimate from an image pair. The training is performed using a semi-supervised approach, combining the unsupervised GAN technique with labeled data. Unlike geometry based monocular methods, the proposed method can recover the absolute scale of the observed scene without neither prior knowledge nor extra information as it can infer it from the training stage. The evaluation of the resulting system is carried out on the well-known KITTI dataset where it is shown to work in real time and the accuracy obtained is encouraging to continue the development of Deep Learning based methods.

### Paper
* **WGANVO: odometría visual monocular basada en redes adversarias generativas**, Javier Cremona, Lucas C. Uzal, Taihú Pire, Revista Iberoamericana de Automática e Informática industrial, [S.l.], dic. 2021. ISSN 1697-7920. Disponible en: [pdf](https://polipapers.upv.es/index.php/RIAI/article/view/16113). DOI: https://doi.org/10.4995/riai.2022.16113.

* **WGANVO: Monocular Visual Odometry based on Generative Adversarial Networks**, Javier Cremona, Lucas C. Uzal, Taihú Pire, arXiv [pdf](https://arxiv.org/abs/2007.13704).

### How to cite
```
@article{RIAI16113,
	author = {Javier Alejandro Cremona and Lucas Uzal and Taihú Pire},
	title = {WGANVO: odometría visual monocular basada en redes adversarias generativas},
	journal = {Revista Iberoamericana de Automática e Informática industrial},
	volume = {0},
	number = {0},
	year = {2021},
	keywords = {Localización; Redes Neuronales; Robots Móviles},
	abstract = {Los sistemas tradicionales de odometría visual (VO), directos o basados en características visuales, son susceptibles de cometer errores de correspondencia entre imágenes. Además, las configuraciones monoculares sólo son capaces de estimar la localización sujeto a un factor de escala, lo que hace imposible su uso inmediato en aplicaciones de robótica o realidad virtual. Recientemente, varios problemas de Visión por Computadora han sido abordados con éxito por algoritmos de Aprendizaje Profundo. En este trabajo presentamos un sistema de odometría visual monocular basado en Aprendizaje Profundo llamado WGANVO. Específicamente, entrenamos una red neuronal basada en GAN para regresionar una estimación de movimiento. El modelo resultante recibe un par de imágenes y estima el movimiento relativo entre ellas. Entrenamos la red neuronal utilizando un enfoque semi-supervisado. A diferencia de los sistemas monoculares tradicionales basados en geometría, nuestro método basado en Deep Learning es capaz de estimar la escala absoluta de la escena sin información extra ni conocimiento previo. Evaluamos WGANVO en el conocido conjunto de datos KITTI. Demostramos que nuestro sistema funciona en tiempo real y la precisión obtenida alienta a seguir desarrollando sistemas de localización basados en Aprendizaje Profundo.},
	issn = {1697-7920},	
	doi = {10.4995/riai.2022.16113},
	url = {https://polipapers.upv.es/index.php/RIAI/article/view/16113}
}
```

## Video 1
<a href="https://www.youtube.com/watch?v=6vcR9PCsWDQ" target="_blank">
  <img src="https://user-images.githubusercontent.com/3181393/260768754-a8ea542e-b36c-4a4d-8021-5418262293fd.png" alt="Demo 1" width="700" />
</a>

## Video 2
<a href="https://www.youtube.com/watch?v=zg5BlvUQhWE" target="_blank">
  <img src="https://user-images.githubusercontent.com/3181393/260743282-d90481df-d0e9-48b7-865f-0d9e021ef2d1.png" alt="Demo 2" width="700" />
</a>

# License 
Our work is released under a [GPLv3 license](License-gpl.txt), except for some files.
The scripts that are used to pre-process the images are released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](License-CCBYNCSA4.txt) (see [LICENSE.txt](LICENSE.txt)). For a list of dependencies (and associated licenses), please see [Dependencies.md](Dependencies.md).

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
**Note**: Folder names (`train_images`, `00`, `01`, `03`) are not required to be the same as the ones in this example.

Then, you must repeat this step in order to generate images to perform the adversarial training and to test the network. After that, copy everything into `images-dir` folder. This folder will be mounted as a volume in the Docker container. For example, you may ended up having this structure.
```
images-dir/
├── train_images/
│   ├── 00
│   ├── 01
│   └── 03
├── train_gan_images/
│   ├── 06
│   ├── 07
│   └── 08
└── test_images/
    └── 04
```
**Note**: Try to have at least 2 folders in `train_images`.

A shell in the Docker container must be opened:
```
make shell
```
Alternatively, you can run `docker run -it --rm --runtime=nvidia -v $(pwd)/images-dir:/var/kitti wganvo_wganvo:latest` or `docker run -it --rm --gpus all -v $(pwd)/images-dir:/var/kitti wganvo_wganvo:latest` in newer versions of Docker. It may be different based on your machine’s operating system and the kind of NVIDIA GPU that your machine has. See [this link](https://towardsdatascience.com/how-to-properly-use-the-gpu-within-a-docker-container-4c699c78c6d1).

The main script is `wgan/wgan_improved.py`. In the container's shell you can run this script to train the network. 
```
python wgan/wgan_improved.py /var/kitti/train_images /var/kitti/test_images /var/kitti/train_gan_images --batch_size <BATCH_SIZE>
```
The command `python wgan/wgan_improved.py -h` will display all the options that can be configured. It is important to set `--log_dir`.

## Testing
In order to test the resulting network, `vgg_trainable/test/test_model.py` can be used. Run:
```
python vgg_trainable/test/test_model.py <MODEL_NAME> /var/kitti/test_images/ --batch_size <BATCH_SIZE> 
```
where `<BATCH_SIZE>` is the batch size used to train the network and `<MODEL_NAME>` is the name of the model that was saved in the log directory (the path was set using `--log_dir`). The name of the model is the name of the file `<MODEL_NAME>.meta` (supply it to `test_model.py` without the `.meta` suffix).

<!--Para correr el test `vgg_trainable/test/test_model.py`, guardar las imágenes y el modelo en `images_dir` buscar donde se creo el volume, y en el shell del Docker, correr el test apuntando al volume. -->
