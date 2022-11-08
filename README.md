# boostcamp lv1 image classification competition 1

### 팀원
| 인덱스 |  이름 | 깃허브 | 이메일 | 역할 |
|:--:|:----:|:----:|:----:|:----:|
| 1  | 김지훈 | <a href="https://github.com/kzh3010">kzh3010</a> | [![logo](https://img.shields.io/badge/Mail-kzh3010@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:kzh3010@gmail.com) | Multi label
| 2  | 원준식 | <a href="https://github.com/JSJSWON">JSJSWON</a> | [![logo](https://img.shields.io/badge/Mail-jswonjswon@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:jswonjswon@gmail.com) | Mask task, Multi label
| 3  | 백우열 | <a href="https://github.com/wooyeolBaek">wooyeolBaek</a> | [![logo](https://img.shields.io/badge/Mail-dwybaek7@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:dwybaek7@gmail.com) | Age task
| 4  | 조용재 | <a href="https://github.com/yyongjae">yyongjae</a> | [![logo](https://img.shields.io/badge/Mail-dydwo706@gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:dydwo706@gmail.com) | Gender task

# 대회 설명(문제)

COVID-19의 확산으로 전 세계 사람들의 활동에 많은 제약이 발생 했습니다.

전파력이 강한 COVID-19의 감염을 막기 위해 사람들은 마스크를 올바르게 착용하여 코와 입을 막아야 합니다.  

본 대회는 넓은 공공장소에서 적은 인적자원을 투입해 사람 얼굴 이미지로 마스크 착용 유무를 파악하기 위한 Project 입니다.

부스트캠프 Level_1 stage 강의 동안 배운 내용을 바탕으로 image classification을 위한 모델 설계, 학습을 진행하고, 그 결과에 따른 순위를 산정하는 방식으로 진행되었습니다.

## 데이터셋 구조
| Class | Mask | Gender | Age |
|:---:|:---:|:---:|:----:|
| 0 | Wear | Male | <30 | 
| 1 | Wear | Male | ≥30 and <60 | 
| 2 | Wear | Male | ≥60  |
| 3 | Wear | Female | <30 |
| 4 | Wear | Female | ≥30 and <60 | 
| 5 | Wear | Female | ≥60 |
| 6 | Incorrect | Male | <30 |
| 7 | Incorrect | Male | ≥30 and <60 | 
| 8 | Incorrect | Male | ≥60 |
| 9 | Incorrect | Female | <30 |
| 10 | Incorrect | Female | ≥30 and <60 | 
| 11 | Incorrect | Female | ≥60 |
| 12 | Not Wear | Male | <30 |
| 13 | Not Wear | Male | ≥30 and <60 | 
| 14 | Not Wear | Male | ≥60 |
| 15 | Not Wear | Female | <30 |
| 16 | Not Wear | Female | ≥30 and <60 | 
| 17 | Not Wear | Female | ≥60 |

# 파이프라인

![앙상블_model_flow.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a2c5cf6f-6a86-4edc-a551-f40951bebd48/%E1%84%8B%E1%85%A1%E1%86%BC%E1%84%89%E1%85%A1%E1%86%BC%E1%84%87%E1%85%B3%E1%86%AF_model_flow.jpg)

# Requirements

```python
torch==1.12.1
torchvision==0.13.1
tensorboard==2.4.1
pandas==1.1.5
opencv-python==4.5.1.48
scikit-learn==1.1.3
matplotlib==3.2.1
wandb==0.13.5
```

# 실행 방법

## 1. 코드 구조

- dataset.py: 데이터에 맞는 sampler와 stratify 기능을 추가한 `CustomDataset` 사용
- loss.py: f1 loss, focal loss, label smoothing loss 같은 따로 구현된 loss functions 저장
- model.py: Backbone으로 사용한 Classification model들 저장
- train.py: CustomDataset에서 데이터를 받아 `5 fold`로 `Cross Validation` 방식으로 학습 진행
- inference.py: `5 fold`로 저장한 모델의 가중치 5개를 `softvoting` 방식으로 inference 수행

## 2. Training

```bash
!python3 [train.py](http://train.py/) \
	--epochs 20 \
	--augmentation CustomAugmentation \
	--model resnet18 \
	--criterion label_smoothing \
	--valid_batch_size 70 \
	--lr_decay_step 5
```

Training 기록을 저장하기 위해 `./model/` 경로에 폴더를 생성합니다.

Cross validation마다 가장 높은 macro-f1 score를 기록한 모델이 위에서 만든 폴더에 `.ckpt` 파일로 저장됩니다.

## 3. Inference

```bash
!python3 [inference.py](http://inference1.py/) \
	--model_dir `폴더명`
```

Training에서 만든 `.ckpt` 파일들을 불러와 inference를 진행합니다.

# 환경

*코드 공유: github*
*서버: AI Stage V100*
*정리: 노션, google docs*
*사용 도구: Python3, Pytorch, WandB, Tensorboard, JupyterLab, Visual Studio Code*
