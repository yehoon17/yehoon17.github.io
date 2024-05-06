---
title: Scikit-learn GridSearchCV에 cv 설정
date: 2024-05-04 05:25:00 +09:00
categories: [데이터 분석]
author: yehoon
tags: [Pandas, Scikit-learn]
---

교차 검증(cross validation): 머신 러닝 모델의 일반화 성능을 평가하는 데 사용되는 방법  

모델을 하나의 데이터셋에서 훈련하고 다른 데이터셋에서 평가하는 대신, 교차 검증은 데이터셋을 여러 하위 집합으로 나누고 이러한 하위 집합 중 일부에서 모델을 훈련한 다음 나머지 하위 집합에서 모델을 평가

- 모델의 성능을 더 신뢰할 수 있는 추정을 제공
-  과적합을 감지하는 데 도움
-  더 나은 모델 선택 및 하이퍼파라미터 튜닝이 가능
-   훈련 및 테스트 목적으로 사용 가능한 데이터의 활용을 극대화


## 기본 cross validation 
```python 
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# iris 데이터셋 불러오기
iris = load_iris()
X = iris.data
y = iris.target

# classifier 설정
clf = DecisionTreeClassifier()

# cross-validation 수행
scores = cross_val_score(clf, X, y, cv=5)  
print("Accuracy:", scores.mean())
```

## StratifiedKFold
각 폴드에서 클래스의 비율을 보존하는 교차 검증 기법
 - 이는 각 폴드가 원본 데이터셋과 유사한 타겟 클래스 분포를 갖도록 보장
 - 특히 클래스 불균형이 있는 데이터셋에 유용

```python 
from sklearn.model_selection import StratifiedKFold

# StratifiedKFold 설정
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# cross-validation 수행
skf_scores = cross_val_score(clf, X, y, cv=skf)
print("StratifiedKFold Accuracy:", skf_scores.mean())

```

## GridSearchCV에 cv 설정
`GridSearchCV`의 `cv` 매개변수는 다음과 같은 다양한 교차 검증 전략이 적용 가능하다.

1. `KFold` 교차 검증의 폴드 수를 지정하는 정수 값
2. 교차 검증 분할기의 인스턴스(e.g., `KFold`, `StratifiedKFold`, `TimeSeriesSplit` 등)
3. 훈련/테스트 분할을 생성하는 교차 검증 이터레이터

```python 
from sklearn.model_selection import GridSearchCV

# hyperparameter grid 설정
param_grid = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# GridSearchCV 설정
grid_search = GridSearchCV(clf, param_grid, cv=skf)

# Fit
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
```
