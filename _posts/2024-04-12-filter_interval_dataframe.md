---
title: 데이터프레임 구간 필터링
date: 2024-04-12 18:25:00 +09:00
categories: [데이터 분석, 전처리]
author: yehoon
tags: [Pandas]
description: 데이터프레임 구간 필터링
---

## 특정 컬럼이 연속적으로 0인 구간 추출하기
```python
def find_zero_intervals_indices(df, col, n=3):
    zero_intervals_indices = []
    current_start_index = None

    for i in range(len(df)):
        if df.loc[i, col] == 0:
            if current_start_index is None:
                current_start_index = i
        else:
            if current_start_index is not None:
                interval_length = i - current_start_index
                if interval_length >= n:
                    zero_intervals_indices.append((current_start_index, i - 1))
                current_start_index = None

    if current_start_index is not None:
        interval_length = len(df) - current_start_index
        if interval_length >= n:
            zero_intervals_indices.append((current_start_index, len(df) - 1))

    return zero_intervals_indices
```

### 예시
```python
data = {'Time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Speed': [10, 0, 0, 0, 20, 0, 0, 0, 0, 30]}
df = pd.DataFrame(data)

intervals_indices = find_zero_intervals_indices(df, 'Speed')

for interval in intervals_indices:
    print(interval)
```

**출력:**    
(1, 3)  
(5, 8)

```python
start, end = interval
df[start: end+1]
```

**출력:**    

|    |   Time |   Speed |
|---:|-------:|--------:|
|  5 |      6 |       0 |
|  6 |      7 |       0 |
|  7 |      8 |       0 |
|  8 |      9 |       0 |


## 특정 컬림이 연속적으로 증가하는 구간 추출하기
```python 
def find_increasing_intervals_indices(df, col, n):
    increasing_intervals_indices = []
    current_start_index = None

    for i in range(len(df) - 1):
        if df.loc[i+1, col] > df.loc[i, col]:
            if current_start_index is None:
                current_start_index = i
        else:
            if current_start_index is not None:
                interval_length = i - current_start_index + 1
                if interval_length >= n:
                    increasing_intervals_indices.append((current_start_index, i))
                current_start_index = None

    if current_start_index is not None:
        interval_length = len(df) - current_start_index
        if interval_length >= n:
            increasing_intervals_indices.append((current_start_index, len(df) - 1))

    return increasing_intervals_indices
```

### 예시
```python
data = {'x': [1, 2, 3, 5, 7, 9, 8, 12, 15, 18]}
df = pd.DataFrame(data)

intervals_indices = find_increasing_intervals_indices(df, 'x', 3)

for interval in intervals_indices:
    print(interval)
```

**출력:**    
(0, 5)  
(6, 9)

## 구간으로 데이터프레임 분할하기
```python
def split_intervals(df, intervals_indices):
    interval_dfs = []
    non_interval_dfs = []

    start_index = 0
    for start, end in intervals_indices:
        if start_index < start:
            non_interval_dfs.append(df.iloc[start_index:start])

        interval_dfs.append(df.iloc[start:end+1])
        start_index = end + 1

    if start_index < len(df):
        non_interval_dfs.append(df.iloc[start_index:])

    return interval_dfs, non_interval_dfs
```

### 예시
```python
data = {'x': [1, 2, 3, 5, 7, 9, 8, 12, 15, 18]}
df = pd.DataFrame(data)

intervals_indices = [(0, 3), (7, 9)]
interval_dfs, non_interval_dfs = split_intervals(df, intervals_indices)

print("Intervals:")
for interval_df in interval_dfs:
    print(interval_df.to_markdown())
    print("="*20)

print("Non-Intervals:")
for non_interval_df in non_interval_dfs:
    print(non_interval_df.to_markdown())
    print("="*20)
```

**출력:**    
Intervals:   

|    |   x |
|---:|----:|
|  0 |   1 |
|  1 |   2 |
|  2 |   3 |
|  3 |   5 |

====================   

|    |   x |
|---:|----:|
|  7 |  12 |
|  8 |  15 |
|  9 |  18 |

====================   
Non-Intervals:

|    |   x |
|---:|----:|
|  4 |   7 |
|  5 |   9 |
|  6 |   8 |

====================   
