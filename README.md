## TransGOP: Transformer-based Gaze Object Prediction
This repository is the official implementation of TransGOP, which studies the gaze object prediction task.
In this work, we introduces Transformer into the fields of gaze object prediction and proposes an end-to-end Transformer-based gaze object prediction method named TransGOP. Specifically, TransGOP uses of-the-shelf Transformer-based object detector to detect the location of objects, and fed the fused feature of the head image and scene image into a Transformer-based gaze prediction branch to regress the gaze heatmap. Moreover, to better enhance the gaze regressor by the informative knowledge from the object detectors, we propose an object-to-gaze attention mechanism to let the queries in the gaze regressor receive the encoded features of the object detector. Finally, to make the whole framework can be end-to-end trained, we propose a Gaze Box loss to jointly optimize the object detector and gaze regressor by enhancing the gaze energy in the box of the stared object. Extensive experiments on the GOO-Synth and GOO-Real datasets demonstrate that our TransGOP achieves state-of-the-art performance on all tracks, i.e. , object detection, gaze estimation, and gaze object prediction.
![Illustrating the architecture of the proposed TransGOP](./figs/fig_frame.jpg)

## Data Preparation
The GOO dataset contains two subsets: GOO-Sync and GOO-Real. 

You can download GOO-Synth dataset and annotations from Baidu Netdisk:

[GOOsynth-data and annotations](https://pan.baidu.com/s/1pe5kj9z3mFPl0guatPVrvA)(code:166j)



You can download GOO-Real dataset and annotations from Baidu Netdisk:

[GOOreal-data and annotations](https://pan.baidu.com/s/1Flfs15vBaCeuST5a5zkQiA)(code:pfni)



~~~~
Please ensure the data structure is as below

├── Datasets
   └── goosynth
       └── annotations
          ├── gop_train.json
          ├── gop_val.json
       └── val
          ├── 0.png
          ├── 1.png
          ├── ...
       └── train
          ├── 0.png
          ├── 1.png
          ├── ...
   └── gooreal
        └── annotations
          ├── gop_train.json
          ├── gop_val.json
        └── val
          ├── 0.png
          ├── 1.png
          ├── ...
        └── train
          ├── 0.png
          ├── 1.png
          ├── ...
~~~~

Environment Preparation


```
conda env create -n TransGOP -f environment.yaml
```

Compiling CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   ```

## Training & Inference

To carry out experiments on the GOO dataset, please follow these commands:

Experiments on GOO-Synth:
```sh
bash scripts/TransGOP_train.sh /Dateses/goosynth/
```
Experiments on GOO-Real:
```sh
bash scripts/TransGOP_train.sh /Dateses/gooreal/
```

## Pre-trained Models
You can download pretrained models from baiduyun:

[Pre-trained Models on GOO-Real dataset](https://pan.baidu.com/s/1atXlAI93C1e8yzx0bqwcnA) (code:e570). 

## Get_Result
Test on the GOO-Synth:

  ```sh
  bash scripts/TransGOP_eval.sh /Dateses/goosynth/ /path/to/your/checkpoint
  ```
Test on the GOO-Real:

  ```sh
  bash scripts/TransGOP_eval.sh /Dateses/gooreal/ /path/to/your/checkpoint
  ```

## Results

Our model achieves the following performance on GOOSynth dataset:

|  AUC  | Dist. | Ang.  |  AP  | AP50 | AP75 | Gaze object prediction mSoC (%) |
| :---: | :---: | :---: | :--: | :--: | :--: | :-----------------------------: |
| 0.963 | 0.079 | 13.30 | 87.6 | 99.0 | 97.3 |              92.8               |

