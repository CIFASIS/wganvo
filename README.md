WGANVO
=============================

Requirements
------------
Instalar Docker y docker-compose.


Utilizando el modelo entrenado
------------------------------
```
sudo make image
sudo make start
sudo make shell
```

Para correr el test `vgg_trainable/test/test_model.py`, guardar las imágenes y el modelo en `images_dir` buscar donde se creo el volume, y en el shell del Docker, correr el test apuntando al volume.



