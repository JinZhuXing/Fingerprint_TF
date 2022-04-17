# Fingerprint_TF

[![Join the chat at https://gitter.im/Fingerprint_TF/community](https://badges.gitter.im/Fingerprint_TF/community.svg)](https://gitter.im/Fingerprint_TF/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Deep Learning fingerprint recognition using Tensorflow2.

### Image Information
    Fingerprint image size is 160x160(500DPI).

### Environment
    Python : 3.7

    Tensorflow : >= 2.0

### Reference
    https://github.com/kairess/fingerprint_recognition

### Sample Dataset
You can get sample dataset from here.

---

[Fingerprint Dataset for FVC2000_DB4_B in BaiduYun, Password: xeqp](https://pan.baidu.com/s/1Y4L5xgHNK47BeR9bVbRrqA)

[Fingerprint Dataset for FVC2000_DB4_B in Google Drive](https://drive.google.com/file/d/1ArZ9ECb9fCDW0UUGzeugaeMBT3_FD7U0/view?usp=sharing)

[Fingerprint Dataset for FVC2000_DB4_B in Kaggle](https://www.kaggle.com/peace1019/fingerprint-dataset-for-fvc2000-db4-b)

---

This dataset was created from FVC2000_DB4_B.


### Pretrained Model
| Release Date | Model Version | Training Dataset | Image Information | Validation accuracy | URL |
| :----: | :----: | :----: | :----: | :----: | :----: |
| 2020-02-21 | v0.1 | 17,859 Images (343 Fingers) | 160 x 160 (500DPI) | 0.9702 | [model.7z](https://github.com/JinZhuXing/Fingerprint_TF/releases/download/0.1/model.7z)
| 2020-04-13 | v0.2_Beta1 | 5,800 Images (203 Fingers) | 160 x 160 (500DPI) | 0.9748 | [model_0_2_b1.7z](https://github.com/JinZhuXing/Fingerprint_TF/releases/download/0.2(beta1)/model.7z)


## Preprocess

Before you train model, you must do preprocess.

Preprocess can generate numpy data from original image.

Please see dataset/README.md file.


## Train

You can train your model using source/02_train source.

### Model Structure

![Alt text](resource/Model_Structure.png "optional title")


## Evaluation

You can evaluate your model using source/03_evaluation source. 


## Donate
Please donate fingerprint image dataset for research purpose.

Thanks.
