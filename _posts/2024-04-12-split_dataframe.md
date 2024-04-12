---
title: 데이터프레임 분할하기
date: 2024-04-12 17:25:00 +09:00
categories: [데이터 분석, 전처리]
author: yehoon
tags: [Pandas]
description: 데이터프레임 분할하기
---


## 데이터프레임을 길이 또는 개수로 분할하기
```python
def split_dataframe(df, size=None, num=None):
    if size is None and num is None:
        raise ValueError("Either 'size' or 'num' must be provided.")
    elif size is not None and num is not None:
        raise ValueError("Only one of 'size' or 'num' should be provided.")
    
    if size is not None:
        num = -(-len(df) // size)  
    
    chunk_size = len(df) // num
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num)]
    
    remainder = len(df) % num
    if remainder:
        last_chunk = df.iloc[-remainder:]
        chunks.append(last_chunk)
    
    return chunks
```

## 데이터프레임을 임의의 길이로 분할하기
```python 
import random

def split_dataframe_within_range(df, a, b, random_state=None):
    if not isinstance(a, int) or not isinstance(b, int) or a <= 0 or b <= 0:
        raise ValueError("Sizes 'a' and 'b' must be positive integers.")
    if a > b:
        raise ValueError("'a' must be less than or equal to 'b'.")

    total_size = len(df)
    chunks = []
    random.seed(random_state)
    while total_size > 0:
        size = random.randint(a, b)  
        if size > total_size:
            size = total_size  
        chunks.append(df.iloc[:size])  
        df = df.iloc[size:] 
        total_size -= size  
    return chunks
```

## 여러 데이터프레임을 분할하기
```python 
def split_dataframes_list(dataframes, size=None, num=None):
    split_dataframes = []
    for df in dataframes:
        split_dataframes.extend(split_dataframe(df, size=size, num=num))
    return split_dataframes
```

## (메모용) 여러 데이터프레임의 마지막 값과 처음 값의 차이로 새로운 데이터프레임 만들기
```python 
def get_row_differences(dataframes):
    differences = []
    for df in dataframes:
        if len(df) < 2:
            raise ValueError("Each DataFrame must have at least two rows.")
        first_row = df.iloc[0]
        last_row = df.iloc[-1]
        difference = last_row - first_row
        differences.append(difference)
    return pd.DataFrame(differences)
```

