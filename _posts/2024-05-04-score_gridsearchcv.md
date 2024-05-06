---
title: Scikit-learn GridSearchCV에 여러 score 설정
date: 2024-05-04 05:25:00 +09:00
categories: [데이터 분석]
author: yehoon
tags: [Pandas, Scikit-learn]
---

Scikit-learn 모델을 훈련하면서 여러 score에 대해 확인하려고 한다.  
GridSearchCV를 설정해서 이를 쉽게 구현할 수 있다.

## 예시 Classification
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# iris 데이터셋 불러오기
X, y = load_iris(return_X_y=True)

# 훈련, 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# classifier 설정
clf = RandomForestClassifier()

# parameter grid 설정
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}

# scoring metrics 설정
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_macro',
    'recall': 'recall_macro',
    'f1_score': 'f1_macro'
}

# grid search 수행
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy')
grid_search.fit(X_train, y_train)

# best model and parameters 확인
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 예측
y_pred = best_model.predict(X_test)

# metrics 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print('Best Parameters:', best_params)
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1)
```

Scikit-learn scoring: <https://scikit-learn.org/stable/modules/model_evaluation.html>

 - dictionary `scoring`에 `'이름': 'Scikit-learn scoring'` 지정
 - `GridSearchCV`에 `scoring=scoring` 전달하고, `refit='Scikit-learn scoring'` 설정

## Metrics

### 분류 
1. **정확도(Accuracy)**:
   - 전체 인스턴스 중 올바르게 예측된 인스턴스의 비율을 측정
   - 균형 잡힌 데이터셋에 적합하지만 불균형 데이터셋에서는 주의 필요

2. **정밀도(Precision)**:
   - 모든 양성 예측 중 실제 양성 예측의 비율을 측정
   - 분류기가 음성 샘플을 양성으로 분류하지 않는 능력을 표현

3. **재현율 (Recall)**:
   - 모든 실제 양성 인스턴스 중 실제 양성 예측의 비율을 측정
   - 분류기가 모든 양성 인스턴스를 찾는 능력을 표현

4. **F1-score**:
   - 정밀도와 재현율의 조화 평균
   - 특히 불균형 데이터셋에 대해 정밀도와 재현율 사이의 균형을 제공

5. **ROC-AUC** (Receiver Operating Characteristic - Area Under the Curve):
   - ROC 곡선 아래 영역을 측정
   - ROC 곡선은 진짜 양성 비율 대 거짓 양성 비율을 표현
   - 분류기가 클래스를 구분하는 능력을 표현
   - 이진 분류 작업에 적합

6. **로그 손실(Log Loss)**:
   - 예측 출력이 0과 1 사이의 확률 값인 분류 모델의 성능을 측정
   - 확신이 있을 때 잘못된 예측을 보다 강하게 벌함

#### Macro, Micro
분류 metric의 macro와 micro는 다중 클래스 분류 문제를 다룰 때 정밀도, 재현율 및 F1-점수 등을 계산하는 데 사용되는 평균화 방법

- **매크로 평균**: 클래스 불균형 여부에 관계없이 모든 클래스를 동등하게 취급
- **마이크로 평균**: 인스턴스가 많은 클래스에 더 많은 가중치를 부여하며, 불균형 데이터셋에 적합



### 회귀 
1. **평균 제곱 오차 (MSE)**:
   - MSE는 오차의 제곱의 평균으로 예측 값과 실제 값 사이의 평균 제곱 차이를 측정
   - 큰 오류에 더 많은 가중치를 부여

2. **평균 절대 오차 (MAE)**:
   - MAE는 절대 오차의 평균으로 예측 값과 실제 값 사이의 평균 절대 차이를 측정
   - MSE에 비해 이상치에 덜 민감

3. **R^2 (결정 계수)**:
   - R^2는 종속 변수의 변동성 중 독립 변수로부터 예측 가능한 변동성의 비율을 측정
   - 1은 완벽한 적합을 나타냄




