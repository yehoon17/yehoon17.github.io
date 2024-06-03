---
title: Attention 정리
date: 2024-06-03 13:00:00 +09:00
categories: [Machine Learning]
author: yehoon
tags: [Attention]
math: true

---

## Attention이란?
신경망이 입력 데이터의 특정 부분에 더 집중하게 하는 메커니즘

## 사전 지식
- 인코더 (Encoder)
  -  입력 시퀀스를 처리하여 고정 차원의 표현으로 변환하는 seq2seq 모델의 구성 요소
  -  이 표현은 입력 시퀀스의 정보를 캡처하여 디코더에서 사용됨
- 디코더 (Decoder)
  - 인코더가 제공한 정보와 컨텍스트 벡터를 기반으로 출력 시퀀스를 생성하는 seq2seq 모델의 구성 요소
- Alignment
  - 정렬은 입력 시퀀스와 출력 시퀀스의 요소 간의 대응 관계
- 컨텍스트 벡터 (Context Vector)
  - 특정 시점에서 입력 시퀀스의 주목된 정보를 요약한 벡터
    - 어텐션 메커니즘의 가중치를 기반으로 입력 데이터의 값들을 가중 합
  - 모델이 현재 시점에서 출력을 생성할 때 필요한 컨텍스트나 관련 정보를 제공


## 배경
- 인간이 정보를 처리할때, 특정 부분에 집중하는 메커니즘을 도입
- RNN의 한계 극복
  - 기울기 소실 문제(Gradient Vanishing Problem)로 인해 성능이 저하

## 메커니즘

![alt text](/assets/img/attention/architecture.png)

1. **쿼리(Query), 키(Key), 값(Value) 벡터 생성**
   - 인코더의 출력(hidden state) 벡터를 Key와 Value 벡터로 사용
   - 디코더의 현재 상태(hidden state)를 Query 벡터로 사용

2. **어텐션 스코어 계산**
   - 각 Query 벡터와 모든 Key 벡터 사이의 유사성을 계산하여 어텐션 스코어 도출

3. **어텐션 가중치 계산**:
   - 어텐션 스코어를 softmax 함수로 정규화하여 어텐션 가중치 계산
   - 이 가중치는 각 Key-Value 쌍이 얼마나 중요한지를 표현

     $$
     \alpha_i = \frac{\exp(\text{Attention Score}(q, k_i))}{\sum_{j} \exp(\text{Attention Score}(q, k_j))}
     $$

4. **컨텍스트 벡터 생성**:
   - 어텐션 가중치를 각 Value 벡터에 곱한 후, 이를 합산하여 컨텍스트 벡터를 생성

     $$
     \text{Context Vector} = \sum_{i} \alpha_i v_i
     $$

5. **디코더에서 출력 생성**:
   - 생성된 컨텍스트 벡터는 디코더의 입력으로 사용되어, 디코더가 다음 출력 토큰을 예측하는 데 사용

## 어텐션 스코어 종류
![_](/assets/img/attention/score_functions.png)

## 어텐션 메커니즘 분류
![_](/assets/img/attention/category.png)

### Softness
#### 소프트 어텐션 (Soft Attention)
- 모든 입력 요소에 연속적인 어텐션 가중치를 할당
- 입력의 모든 부분이 중요도에 따라 가중치를 부여
- 미분 가능하여 학습이 용이

#### 하드 어텐션 (Hard Attention)
- 입력의 특정 부분에 이산적인 결정을 내려 어텐션을 집중
- 선택된 입력 요소만 집중
- 계산 효율성이 높지만, 학습이 어려울 수 있음

#### 글로벌 어텐션 (Global Attention)
- 전체 입력 시퀀스를 대상으로 어텐션을 분배
- 소프트 어텐션과 유사

#### 로컬 어텐션 (Local Attention)
- 입력의 특정 부분(윈도우)에만 어텐션을 집중
- 입력의 일부 영역만 집중적으로 처리
- 계산 비용 절감과 특정 위치에 대한 집중이 가능

### Input Representations
#### Distinctive
- Key와 Query는 두 개의 별도 시퀀스에서 생성
- 단일 입력 시퀀스에서 작동하고 해당하는 출력 시퀀스를 생성

#### Self-attention
- 입력 시퀀스 자체만을 기반으로 어텐션를 계산
- Query, Key, Value가 모두 동일한 입력 시퀀스의 다른 표현

#### Co-attention
- 다중 입력에 대해 어텐션을 처리하는 메커니즘
- 여러 모달리티를 다루는 작업에 유용

#### Hierarchical
- 문서나 이미지 등의 입력에서 다중 수준의 어텐션을 계산하여 추론하는 메커니즘

### Output Representations
![_](/assets/img/attention/multi.png)

#### Multi-head
- 여러 (Query, Key, Value)를 생성하고, 각각 계산한 후에 concat

$$
\begin{equation}  
\begin{split}
& A_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right) \\
& O_i = A_iV_i \\
& \text{MultiHead}(Q, K, V) = \text{Concat}(O_1, O_2, ..., O_H) \\
\end{split}
\end{equation}  
$$

#### Multi-dimensional
 - 가중치 점수 벡터 대신 행렬을 사용하여 키에 대한 특징별 점수 벡터를 계산


## 참고
<https://www.sciencedirect.com/science/article/abs/pii/S092523122100477X>  
<https://wikidocs.net/22893>  
<https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html>  
