---
title: 하이퍼파라미터 튜닝 종류
date: 2024-05-07 15:25:00 +09:00
categories: [데이터 분석]
author: yehoon
tags: [Pandas, Scikit-learn, ]
---

하이퍼파라미터 튜닝: 머신러닝 알고리즘의 하이퍼파라미터를 최적화하여 성능과 일반화 능력을 향상시키는 과정

종류:
 - GridSearchCV 
 - RandomizedSearchCV
 - Bayesian Optimization
 - Hyperband
 - 유전 알고리즘

## GridSearchCV
지정된 하이퍼파라미터 하위 집합을 탐색하는 방법
작동 방식:
1. 하이퍼파라미터 그리드 정의
2. 교차 검증
3. 그리드 탐색
4. 최상의 모델 선택
5. 최종 평가

```python 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Iris 데이터셋 불러오기
iris = load_iris()
X = iris.data
y = iris.target

# 훈련, 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# hyperparameter grid 설정
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly']
}

# SVM classifier 생성
svm = SVC()

# GridSearchCV 생성
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')

# 훈련 데이터 적용
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-validation Score:", grid_search.best_score_)

# 최상의 모델 선택
best_model = grid_search.best_estimator_

# 모델 평가
test_score = best_model.score(X_test, y_test)
print("Test Set Score:", test_score)
```



## RandomizedSearchCV

**RandomizedSearchCV**는 GridSearchCV와 같이 모든 가능한 하이퍼파라미터 조합을 체계적으로 탐색하는 대신 **지정된 분포에서 일정 수의 하이퍼파라미터 설정을 샘플링하는 기술**이다.   
이 접근 방식은 하이퍼파라미터 공간이 크고 체계적으로 탐색하기 어려울 때 특히 유용하다.  
무작위 샘플링을 통해 RandomizedSearchCV는 효율적으로 공간을 탐색하고 최적의 하이퍼파라미터가 위치할 수 있는 유망한 영역을 식별할 수 있다.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

# hyperparameter grid 설정
param_dist = {
    'n_estimators': randint(10, 1000),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Random Forest classifier 생성
rf = RandomForestClassifier()

# RandomizedSearchCV 생성
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# 훈련 데이터 적용
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print("Best Parameters:", best_params)

best_model = random_search.best_estimator_
best_model_score = best_model.score(X_test, y_test)
print("Best Model Score:", best_model_score)
```

## Bayesian Optimization
베이지안 최적화(Bayesian Optimization)는 순차적인 모델 기반 최적화 기술로, 다음으로 평가할 하이퍼파라미터 세트를 결정하기 위해 확률적 모델을 사용한다.  
이는 목적 함수(모델 성능)의 surrogate probabilistic model을 구축하고 이를 사용하여 다음으로 시도할 하이퍼파라미터를 결정한다.  
베이지안 최적화는 이전 평가의 결과를 기반으로 목적 함수에 대한 이해를 반복적으로 개선함으로써 더 적은 평가로 최적의 하이퍼파라미터로 빠르게 수렴할 수 있다. 

```python
from skopt import BayesSearchCV
from sklearn.svm import SVC

# parameter space 설정
param_space = {
    'C': (1e-6, 1e+6, 'log-uniform'),
    'gamma': (1e-6, 1e+1, 'log-uniform'),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# SVM classifier 생성
svm = SVC()

# BayesSearchCV 생성
bayes_search = BayesSearchCV(estimator=svm, search_spaces=param_space, n_iter=20, cv=5, verbose=2, n_jobs=-1)

# 훈련 데이터 적용
bayes_search.fit(X_train, y_train)

best_params = bayes_search.best_params_
print("Best Parameters:", best_params)

best_model = bayes_search.best_estimator_
best_model_score = best_model.score(X_test, y_test)
print("Best Model Score:", best_model_score)
```

