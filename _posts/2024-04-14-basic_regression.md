---
title: 기본적인 회귀
date: 2024-04-13 17:25:00 +09:00
categories: [데이터 분석]
author: yehoon
tags: [Pandas, Scikit-learn]
description: 기본적인 회귀
---

## 기본 회귀
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import mean_squared_error


X, y = load_diabetes(return_X_y=True, as_frame=True)

# 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 파이프라인 생성
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', LinearRegression())
])

# 하이퍼파라미터 설정
param_grid = [
    {
        'regression': [LinearRegression()],
        'scaler': [StandardScaler(), MinMaxScaler()],
    },
    {
        'regression': [Ridge()],
        'regression__alpha': [0.1, 1.0, 10.0],
        'scaler': [StandardScaler(), MinMaxScaler()],
    },
    {
        'regression': [Lasso()],
        'regression__alpha': [0.1, 1.0, 10.0],
        'scaler': [StandardScaler(), MinMaxScaler()],
    },
    {
        'regression': [RandomForestRegressor()],
        'regression__n_estimators': [50, 100, 200],
        'regression__max_depth': [None, 10, 20],
        'scaler': [StandardScaler(), MinMaxScaler()],
    }
]


# Grid search 생성
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# 훈련 데이터 적용
grid_search.fit(X_train, y_train)

# 최고 선택
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# 예측
y_pred = best_model.predict(X_test)

# 결과
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
```

