# Tensorflow-OD-LSTM

Repository voor het trainen van een object detector met Bottleneck-LSTM. We vergelijken dit later met het [baseline](https://github.com/LeenGadisseur/Tensorflow-OD-API-workspace) model zonder Bottleneck-LSTM.

Requirements 
------------ 
* TensorFlow 1.14.0 (gpu)
* Python 3.7
* Cuda 10.0
* CuDNN 7

Installatie Object Detection API en 
-----------
De installatie van de Tensorflow object detection API kan je op deze
[link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html) volgen. 
Bij installatie van de [Object detection API](https://github.com/tensorflow/models) gebruik maken van de 
research/object_detection/packages/tf1/setup.py aangezien we werken met TensorFlow 1.14.0.

Indien error voor het importeren van resnet, zet import in commentaar in:
anaconda3/env/tf1/lib/python3.7/site-packages/object_detection/meta_architectures/deepmac_meta_arch.py


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
* [TensorFlow Models](https://github.com/tensorflow/models)
* [EPFL TFRecords](https://drive.google.com/drive/folders/148Ss13RS61af6KCZPEoF1SHUKJAEiDz9?usp=sharing)
* [Baseline](https://github.com/LeenGadisseur/Tensorflow-OD-API-workspace)

