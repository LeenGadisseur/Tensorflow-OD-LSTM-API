# Tensorflow-OD-LSTM

Dit is de repository voor het trainen van een object detector met Bottleneck-LSTM. We vergelijken dit later met het [baseline](https://github.com/LeenGadisseur/Tensorflow-OD-API-workspace) model zonder Bottleneck-LSTM.

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


Gebruik van files
------------------
Het gebruik van de volgende files gebeurt steeds binnen de conda environment.

### Training en evaluatie van ssd_mobilenet_v1_lstm

Commando voor trainen ssd_mobilenet_v1_lstm.
```
python train.py --logtostderr \
	--train_dir=models/my_ssd_mobilenet_v1_lstm \
	--pipeline_config_path=models/my_ssd_mobilenet_v1_lstm/pipeline_shards.config 

```

Commando voor evaluatie van ssd_mobilenet_v1_lstm.
```
python eval.py \
        --logtostderr \
        --checkpoint_dir=models/my_ssd_mobilenet_v1_lstm/checkpoints/ckpt-96x96-b4-5k+7k/model.ckpt-7000 \
        --eval_dir=models/my_ssd_mobilenet_v1_lstm/checkpoints/ckpt-96x96-b4-5k+7k/eval \
        --pipeline_config_path=models/my_ssd_mobilenet_v1_lstm/pipeline_shards.config 

```


### Training en evaluatie van ssd_mobilenet_v2_lstm

Commando voor trainen ssd_mobilenet_v2_lstm.
```
python train.py --logtostderr \
	--train_dir=models/my_ssd_mobilenet_v2_lstm \
	--pipeline_config_path=models/my_ssd_mobilenet_v2_lstm/pipeline_shards.config 

```

Commando voor evaluatie van ssd_mobilenet_v2_lstm.
```
python eval.py \
        --logtostderr \
        --checkpoint_dir=models/my_ssd_mobilenet_v2_lstm/checkpoints/.../model.ckpt \
        --eval_dir=models/my_ssd_mobilenet_v2_lstm/checkpoints/.../eval \
        --pipeline_config_path=models/my_ssd_mobilenet_v2_lstm/pipeline_shards.config 

```



### Training en evaluatie van ssd_mobilenet_v2_interleaved_lstm

Commando voor trainen ssd_mobilenet_v2_interleaved_lstm.
```
python train.py --logtostderr \
	--train_dir=models/my_ssd_mobilenet_v2_interleaved_lstm \
	--pipeline_config_path=models/my_ssd_mobilenet_v2_interleaved_lstm/pipeline_shards.config 

```

Commando voor evaluatie van ssd_mobilenet_v2_interleaved_lstm.
```
python eval.py \
        --logtostderr \
        --checkpoint_dir=models/my_ssd_mobilenet_v1_lstm/.../model.ckpt \
        --eval_dir=models/my_ssd_mobilenet_v1_lstm/.../eval \
        --pipeline_config_path=models/my_ssd_mobilenet_v2_interleaved_lstm/pipeline_shards.config 

```


### Extraheren van tflite modellen

Deze modellen kunnen omgezet worden naar TFLite modellen, dit kan als volgt:

1. Extraheren van een graph file.
```
python export_tflite_lstd_graph.py \
    --pipeline_config_path=models/my_ssd_mobilenet_v1_lstm/pipeline_shards.config \
    --trained_checkpoint_prefix=models/my_ssd_mobilenet_v1_lstm/checkpoints/ckpt-96x96-b4-5k+7k/model.ckpt-7000 \
    --output_directory=models/my_ssd_mobilenet_v1_lstm_tflite/checkpoints/ckpt-96x96-b4-5k+7k

```

2. Omzetten naar een model.tflite file.
```
python export_tflite_lstd_model.py \
    --export_path=models/my_ssd_mobilenet_v1_lstm_tflite/checkpoints/ckpt-96x96-b4-5k+7k/ \
    --pipeline_config_path=models/my_ssd_mobilenet_v1_lstm/pipeline_shards.config \
    --frozen_graph_path=models/my_ssd_mobilenet_v1_lstm_tflite/checkpoints/ckpt-96x96-b4-5k+7k/tflite_graph.pb \
    --resolution=96

```

Het testen van deze modellen geeft momenteel fout : 
RuntimeError: tensorflow/lite/kernels/detection_postprocess.cc:268 input_box_encodings->dims->data[0] != kBatchSize (4 != 1)Node number 138 (TFLite_Detection_PostProcess) failed to invoke.

Links
-----
* [TensorFlow Models](https://github.com/tensorflow/models)
* [EPFL TFRecords](https://drive.google.com/drive/folders/148Ss13RS61af6KCZPEoF1SHUKJAEiDz9?usp=sharing) Momenteel zijn dit nog de TFRecords die gegroepeert zijn per 10 frames, nog vervangen door groepering van 4.
* [Model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md) Pre-trained weights voor de feature extractor zijn hier te vinden.
* [Baseline](https://github.com/LeenGadisseur/Tensorflow-OD-API-workspace)
