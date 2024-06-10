---
title: Batch Normalization
date: 2024-06-10 11:00:00 +09:00
categories: [Machine Learning]
author: yehoon
tags: [Deep Learning]
math: true
---

배치 정규화(batch normalization)이란?
 - 딥러닝 신경망을 훈련할 때 성능과 안정성을 향상시키기 위해 사용하는 기법
 - 각 층의 입력을 정규화하여 평균이 0이고 분산이 1이 되도록 하는 것
   - internal covariate shift 문제 완화

> internal covariate shift: 신경망의 층에 대한 입력 분포가 훈련 중에 변화하는 현상

### Batch Normalization의 작동 원리

1. **정규화**: 각 미니배치에 대해 activation의 평균과 분산을 계산
  
   $$
   \mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
   $$

   $$
   \sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
   $$

    $m$: 미니배치의 예제 수  
    $x_i$: $i$번째 예제의 activation

2. **스케일링과 이동**: activation을 정규화한 후 학습 가능한 매개변수 $\gamma$와 $\beta$를 사용하여 스케일링하고 이동시킴
   
   $$
   \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
   $$

   $$
   y_i = \gamma \hat{x}_i + \beta
   $$

   $\epsilon$: 수치적 안정성을 위해 추가되는 작은 상수(분모 0 방지)

3. **모델에 포함**: 정규화되고 스케일링 및 이동된 활성화 값 $y_i$는 네트워크의 다음 층으로 전달

### Batch Normalization의 이점

1. **internal covariate shift 완화**
   - 각 층에 대한 입력을 정규화함으로써 훈련 과정 내내 안정적인 활성화 분포 유지

2. **빠른 훈련**
   - Batch Normalization이 적용된 네트워크는 더 빠르게 수렴할 수 있음
   -  정규화 덕분에 학습률을 높게 설정해도 발산할 위험이 줄어들어 최적화 속도가 빨라짐

3. **Regularization 효과**
   - Batch Normalization은 미니배치 샘플링으로 인해 약간의 노이즈를 도입하는데, 이는 약간의 regularization 효과를 제공하여 드롭아웃과 같은 다른 regularization 기법의 필요성을 줄일 수 있음

4. **개선된 그래디언트 흐름**
   - 활성화 값을 표준화된 범위로 유지함으로써 그래디언트 흐름을 개선하여 그래디언트 소실 또는 폭발 문제 완화

### 고려사항

- **네트워크 내 위치**
  - 컨볼루션 층 후
  - 완전 연결 층 후
  - activation 함수 전
- **훈련 및 추론 단계**
  - 훈련 중에는 미니배치의 통계(평균과 분산)를 계산
  - 추론 중에는 일관성을 위해 평균과 분산의 이동 평균을 사용
    - 추론할 때는 배치 단위로 실행하는 게 아니라 단일 케이스 단위로 실행하기 때문


### 코드 예제

- **TensorFlow**:
  ```python
  import tensorflow as tf
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPooling2D((2, 2)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  ```

- **PyTorch**:
  ```python
  import torch.nn as nn
  class SimpleCNN(nn.Module):
      def __init__(self):
          super(SimpleCNN, self).__init__()
          self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
          self.bn1 = nn.BatchNorm2d(32)
          self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
          self.fc1 = nn.Linear(32 * 13 * 13, 64)
          self.bn2 = nn.BatchNorm1d(64)
          self.fc2 = nn.Linear(64, 10)

      def forward(self, x):
          x = self.pool(F.relu(self.bn1(self.conv1(x))))
          x = x.view(-1, 32 * 13 * 13)
          x = F.relu(self.bn2(self.fc1(x)))
          x = self.fc2(x)
          return x
  ```
