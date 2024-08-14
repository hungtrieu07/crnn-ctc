<!-- <div align="right">
  Language:
    🇺🇸
  <a title="Chinese" href="./README.zh-CN.md">🇨🇳</a>
</div> -->

<div align="center"><a title="" href="https://github.com/zjykzj/crnn-ctc"><img align="center" src="assets/icons/crnn-ctc.svg" alt=""></a></div>

<p align="center">
  «crnn-ctc» implemented CRNN+CTC
<br>
<br>
  <a href="https://github.com/RichardLitt/standard-readme"><img src="https://img.shields.io/badge/standard--readme-OK-green.svg?style=flat-square" alt=""></a>
  <a href="https://conventionalcommits.org"><img src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg" alt=""></a>
  <a href="http://commitizen.github.io/cz-cli/"><img src="https://img.shields.io/badge/commitizen-friendly-brightgreen.svg" alt=""></a>
</p>

|   **Model**   | **ARCH** | **Input Shape** | **GFLOPs** | **Model Size (MB)** | **EMNIST Accuracy (%)** | **Training Data** | **Testing Data** |
|:-------------:|:--------:|:---------------:|:----------:|:-------------------:|:-----------------------:|:-----------------:|:----------------:|
|   **CRNN**    | CONV+GRU |  (1, 32, 160)   |    2.2     |         31          |         98.718          |      100,000      |      5,000       |
| **CRNN_Tiny** | CONV+GRU |  (1, 32, 160)   |    0.1     |         1.7         |         98.278          |      100,000      |      5,000       |

|   **Model**   | **ARCH** | **Input Shape** | **GFLOPs** | **Model Size (MB)** | **ChineseLicensePlate Accuracy (%)** | **Training Data** | **Testing Data** |
|:-------------:|:--------:|:---------------:|:----------:|:-------------------:|:------------------------------------:|:-----------------:|:----------------:|
|   **CRNN**    | CONV+GRU |  (3, 48, 168)   |    4.0     |         58          |                82.384                |      269,621      |     149,002      |
| **CRNN_Tiny** | CONV+GRU |  (3, 48, 168)   |    0.3     |         4.0         |                76.226                |      269,621      |     149,002      |

For each sub-dataset, the model performance as follows:

|   **Model**   | **CCPD2019-Test Accuracy (%)** | **Testing Data** | **CCPD2020-Test Accuracy (%)** | **Testing Data** |
|:-------------:|:------------------------------:|:----------------:|:------------------------------:|:----------------:|
|   **CRNN**    |             81.761             |     141,982      |             93.728             |      5,006       |
| **CRNN_Tiny** |             75.357             |     141,982      |             92.369             |      5,006       |

## Table of Contents

- [Table of Contents](#table-of-contents)
- [News🚀🚀🚀](#news)
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
  - [Train](#train)
  - [Eval](#eval)
  - [Predict](#predict)
- [Maintainers](#maintainers)
- [Thanks](#thanks)
- [Contributing](#contributing)
- [License](#license)

## News🚀🚀🚀

| Version                                                          | Release Date | Major Updates                                                                                                                                                 |
|------------------------------------------------------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [v1.0.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v1.0.0) | 2024/08/04   | Optimize the CRNN architecture while achieving super lightweight **CRNN_Tiny**. <br>In addition, all training scripts support mixed precision training. |
| [v0.3.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.3.0) | 2024/08/03   | Implement models **CRNN_LSTM** and **CRNN_GRU** on datasets EMNIST and ChineseLicensePlate.                                                                           |
| [v0.2.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.2.0) | 2023/10/11   | Support training/evaluation/prediction of CRNN+CTC based on license plate.                                                                                    |
| [v0.1.0](https://github.com/zjykzj/crnn-ctc/releases/tag/v0.1.0) | 2023/10/10   | Support training/evaluation/prediction of CRNN+CTC based on EMNIST digital characters.                                                                        |

## Background

This warehouse aims to better understand and apply CRNN+CTC, and currently achieves digital recognition and license plate recognition

## Installation

```shell
$ pip install -r requirements.txt
```

## Usage

### Train

* ChineseLicensePlate: [Baidu Drive](https://pan.baidu.com/s/1fQh0E9c6Z4satvrEthKevg)(ad7l)

```shell
# EMNIST
$ python3 train_emnist.py ../datasets/emnist/ ./runs/crnn-emnist-b512/ --batch-size 512 --device 0 --not-tiny
# Plate
$ python3 train_plate.py ../datasets/chinese_license_plate/recog/ ./runs/crnn-plate-b512/ --batch-size 512 --device 0 --not-tiny
```

### Eval

```shell
# EMNIST
$ CUDA_VISIBLE_DEVICES=0 python eval_emnist.py crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist-b512-e100.pth', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist-b512-e100.pth
crnn_tiny-emnist-b512-e100 summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Batch:1562 ACC:100.000: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1563/1563 [00:19<00:00, 80.29it/s]
ACC:98.278
# Plate
$ CUDA_VISIBLE_DEVICES=0 python3 eval_plate.py crnn_tiny-plate-b512-e100.pth ../datasets/chinese_license_plate/recog/
args: Namespace(not_tiny=False, only_ccpd2019=False, only_ccpd2020=False, only_others=False, pretrained='crnn_tiny-plate-b512-e100.pth', use_lstm=False, val_root='../datasets/chinese_license_plate/recog/')
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Load test data: 149002
Batch:4656 ACC:90.000: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4657/4657 [01:08<00:00, 67.50it/s]
ACC:76.226
```

### Predict

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_emnist.py crnn_tiny-emnist-b512-e100.pth ../datasets/emnist/ ./runs/predict/emnist/
args: Namespace(not_tiny=False, pretrained='crnn_tiny-emnist-b512-e100.pth', save_dir='./runs/predict/emnist/', use_lstm=False, val_root='../datasets/emnist/')
Loading CRNN pretrained: crnn_tiny-emnist-b512-e100.pth
crnn_tiny-emnist-b512-e100 summary: 22 layers, 427467 parameters, 427467 gradients, 0.1 GFLOPs
Label: [3 8 5 8 5] Pred: [3 8 5 8 5]
Label: [4 8 6 8 0] Pred: [4 8 6 8 0]
Label: [4 6 4 7 0] Pred: [4 6 4 7 0]
Label: [2 3 5 0 7] Pred: [2 3 5 0 7]
Label: [4 7 8 4 6] Pred: [4 7 8 4 6]
Label: [0 1 4 3 6] Pred: [0 1 4 3 6]
```

![](assets/predict/emnist/predict_emnist.jpg)

```shell
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate-b512-e100.pth ./assets/plate/宁A87J92_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/宁A87J92_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 宁A·87J92 - Predict time: 8.4 ms
Save to runs/predict/plate/plate_宁A87J92_0.jpg
$ CUDA_VISIBLE_DEVICES=0 python predict_plate.py crnn_tiny-plate-b512-e100.pth ./assets/plate/川A3X7J1_0.jpg runs/predict/plate/
args: Namespace(image_path='./assets/plate/川A3X7J1_0.jpg', not_tiny=False, pretrained='crnn_tiny-plate-b512-e100.pth', save_dir='runs/predict/plate/', use_lstm=False)
Loading CRNN pretrained: crnn_tiny-plate-b512-e100.pth
crnn_tiny-plate-b512-e100 summary: 22 layers, 1042318 parameters, 1042318 gradients, 0.3 GFLOPs
Pred: 川A·3X7J1 - Predict time: 8.4 ms
Save to runs/predict/plate/plate_川A3X7J1_0.jpg
```

<p align="left"><img src="assets/predict/plate/plate_宁A87J92_0.jpg" height="240"\>  <img src="assets/predict/plate/plate_川A3X7J1_0.jpg" height="240"\></p>

## Maintainers

* zhujian - *Initial work* - [zjykzj](https://github.com/zjykzj)

## Thanks

* [rinabuoy/crnn-ctc-loss-pytorch](https://github.com/rinabuoy/crnn-ctc-loss-pytorch.git)
* [we0091234/crnn_plate_recognition](https://github.com/we0091234/crnn_plate_recognition.git)
* [zjykzj/LPDet](https://github.com/zjykzj/LPDet)

## Contributing

Anyone's participation is welcome! Open an [issue](https://github.com/zjykzj/crnn-ctc/issues) or submit PRs.

Small note:

* Git submission specifications should be complied
  with [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.4/)
* If versioned, please conform to the [Semantic Versioning 2.0.0](https://semver.org) specification
* If editing the README, please conform to the [standard-readme](https://github.com/RichardLitt/standard-readme)
  specification.

## License

[Apache License 2.0](LICENSE) © 2023 zjykzj