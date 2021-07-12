# Tensorflow-OD-LSTM


Requirements 
------------ 
* Tensorflow 1.14.0
* Python 3.7
* Cuda 10.0
* CuDNN 7

Installatie
-----------
De installatie van de Tensorflow object detection API kan je op deze
[link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html) volgen. 

In research directory volgende commando uitvoeren:
``` 
protoc lstm_object_detection/protos/*.proto --python_out=.
```
Om de lstm_object_detection module te kunnen gebruiken, moet deze ook geplaatst worden in de workspace. 


Annotaties dataset
------------------
TF-records staan niet bij in deze repository. Te downloaden van [hier](https://drive.google.com/drive/folders/148Ss13RS61af6KCZPEoF1SHUKJAEiDz9?usp=sharing) en in de annotaties map plaatsen.

Commando voor trainen ssd_mobilenet_v1_lstm binnen conda environment.
```
python train.py --logtostderr --train_dir=models/my_ssd_mobilenet_v1_lstm --pipeline_config_path=models/my_ssd_mobilenet_v1_lstm/pipeline_shards.config
```

Links
-----
* [TensorFlow Models](https://github.com/tensorflow/models/tree/r1.13.0)
* [EPFL TFRecords](https://drive.google.com/drive/folders/148Ss13RS61af6KCZPEoF1SHUKJAEiDz9?usp=sharing)
* [Baseline](https://github.com/LeenGadisseur/Tensorflow-OD-API-workspace)

