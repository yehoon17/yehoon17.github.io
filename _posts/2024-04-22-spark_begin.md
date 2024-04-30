---
title: Apache Spark 입문
date: 2024-04-22 21:00:00 +09:00
categories: [데이터파이프라인]
author: yehoon
tags: [Spark,  Big Data Processing, PySpark]
description: Apache Spark
image: /assets/img/spark/image.png
---

## Apache Spark란?
 - 빅데이터 처리와 분석을 위해 설계된 오픈 소스 분산 컴퓨팅 시스템
   - 클러스터 환경에서 대규모 데이터 세트를 처리하기 위한 통합 엔진을 제공
   - 배치 처리부터 실시간 스트림 처리 및 기계 학습까지 다양하게 활용

## Apache Spark의 주요 특징:

1. **빠른 처리 속도**: 메모리 내 처리 엔진을 사용하여 매우 빠른 데이터 처리 가능

2. **다양한 기능**: 다양한 데이터 처리 작업을 위한 API와 라이브러리 제공

3. **사용 편의성**: Java, Scala, Python 및 R을 지원

4. **내결함성 보장**: 장애 내성을 보장하기 위해 resilient distributed datasets (RDDs) 사용  
> **RDD(Resilient Distributed Dataset)**: 변경할 수 없는 분산된 객체 컬렉션
>  - 클러스터에서 병렬 처리될 수 있음
>  - 장애에서 회복할 수 있는 회복력을 가지고 있음
>  - 여러 노드에 파티션으로 나뉘어 있어 분산
>  - 데이터셋과 유사한 형태를 가지고 있으며, 모든 유형의 요소를 포함할 수 있음
>  - 기존 RDD에서 새로운 RDD를 생성하는 변환(transformations) 및 변환을 실행하고 결과를 반환하는 액션(actions)


5. **확장성**: Spark의 분산 아키텍처는 대규모 하드웨어 클러스터 확장할 수 있어, 데이터 처리 작업을 효율적으로 확장 가능

## 실습
<https://github.com/yehoon17/Spark-with-MovieLens-Dataset>

### 데이터 
[MovieLens](https://grouplens.org/datasets/movielens/)에서 다운로드 

**movies.csv**
```
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),Comedy|Romance
4,Waiting to Exhale (1995),Comedy|Drama|Romance
```

**ratings.csv**
```
userId,movieId,rating,timestamp
1,296,5.0,1147880044
1,306,3.5,1147868817
1,307,5.0,1147868828
1,665,5.0,1147878820
```

### main.py
```python
# Initialize SparkSession
spark = SparkSession.builder \
    .appName("SparkProject") \
    .getOrCreate()

# Load dataset
ratings_df = spark.read.csv(
    "data/ratings.csv", 
    header=True, 
    inferSchema=True
    )
movies_df = spark.read.csv(
    "data/movies.csv", 
    header=True, 
    inferSchema=True
    )

# Analysis
analyze_top_movies(ratings_df, movies_df)
analyze_user_trands(ratings_df)

# Stop SparkSession
spark.stop()
```

PySpark의 `SparkSession`은 Python에서 Apache Spark를 사용하기 위한 시작점이다.  
 - 구조화된 데이터 작업, SQL 쿼리 실행 및 Spark 속성 구성과 같은 작업을 수행하는 통합된 인터페이스를 제공
 - Spark 리소스(예: Executor 및 메모리)를 관리하고 작업 진행 상황을 모니터링하기 위한 Spark UI에 액세스를 제공
 - Spark 애플리케이션의 라이프사이클을 처리하여 초기화에서 종료까지의 과정을 관리
  
### 상위 영화 분석
```python
def analyze_top_movies(ratings_df, movies_df):
    # Join ratings_df with movies_df to get movie titles and genres
    movie_ratings_df = ratings_df.join(movies_df, "movieId")

    # Calculate the average rating and number of ratings for each movie
    movie_stats_df = (
        movie_ratings_df
        .groupBy("movieId", "title", "genres") 
        .agg(
            count("rating").alias("num_ratings"),
            avg("rating").alias("avg_rating")
            )
        )

    # Calculate a weighted average rating based on the number of ratings
    expr_ = "avg_rating * num_ratings / (num_ratings + 10) + 5 * 10 / (num_ratings + 10)"
    weighted_avg_df = movie_stats_df.withColumn("weighted_avg_rating", expr(expr_))
    
    # Order the movies by weighted average rating and number of ratings
    top_movies_df = weighted_avg_df.orderBy(
        col("weighted_avg_rating").desc(),
        col("num_ratings").desc()
        )

    # Display top N movies
    print("Top 10 Movies by Weighted Average Rating and Number of Ratings:")
    cols = ["title", "genres", "weighted_avg_rating", "num_ratings"]
    top_movies_df.select(cols).show(10, truncate=False)
```

**결과**
```
+---------------------------+--------------------------------+-------------------+-----------+
|title                      |genres                          |weighted_avg_rating|num_ratings|
+---------------------------+--------------------------------+-------------------+-----------+
|Lonesome Dove Church (2014)|Western                         |5.0                |3          |
|Sound of Christmas (2016)  |Drama                           |5.0                |3          |
|Borrowed Time (2012)       |Drama                           |5.0                |3          |
|Awaken (2013)              |Drama|Romance|Sci-Fi            |5.0                |3          |
|The Memory Book (2014)     |Drama|Romance                   |5.0                |2          |
|The Ties That Bind (2015)  |(no genres listed)              |5.0                |2          |
|Joy Road (2011)            |Crime|Drama                     |5.0                |2          |
|Genius on Hold (2013)      |(no genres listed)              |5.0                |2          |
|FB: Fighting Beat (2007)   |Action                          |5.0                |2          |
|Windstorm 2 (2015)         |Adventure|Children|Drama|Romance|5.0                |2          |
+---------------------------+--------------------------------+-------------------+-----------+
```

### 사용자별 평점 추이
```python
def analyze_user_trands(ratings_df):
    # Convert timestamp to a timestamp type
    ratings_df = ratings_df.withColumn("timestamp", from_unixtime("timestamp"))

    # Group ratings by user and time window
    user_trends_df = (
        ratings_df
        .groupBy("userId", window("timestamp", "1 week"))
        .avg("rating")
    )

    # Order the user trends by userId and window
    user_trends_df = user_trends_df.orderBy("userId", "window")

    # Display user trends
    print("User Rating Trends:")
    user_trends_df.show(truncate=False)
```

**결과**
```
+------+------------------------------------------+------------------+
|userId|window                                    |avg(rating)       |
+------+------------------------------------------+------------------+
|1     |{2006-05-11 09:00:00, 2006-05-18 09:00:00}|3.8142857142857145|
|2     |{2006-03-02 09:00:00, 2006-03-09 09:00:00}|3.630434782608696 |
|3     |{2015-08-13 09:00:00, 2015-08-20 09:00:00}|3.7450248756218905|
|3     |{2016-01-21 09:00:00, 2016-01-28 09:00:00}|3.769230769230769 |
|3     |{2017-01-12 09:00:00, 2017-01-19 09:00:00}|3.734375          |
|3     |{2017-04-20 09:00:00, 2017-04-27 09:00:00}|4.055555555555555 |
|3     |{2019-08-15 09:00:00, 2019-08-22 09:00:00}|3.5258064516129033|
|4     |{2019-11-14 09:00:00, 2019-11-21 09:00:00}|3.378099173553719 |
|5     |{1996-04-25 09:00:00, 1996-05-02 09:00:00}|3.6216216216216215|
|5     |{1996-05-09 09:00:00, 1996-05-16 09:00:00}|4.5               |
|5     |{1996-05-23 09:00:00, 1996-05-30 09:00:00}|4.0               |
|5     |{1996-06-20 09:00:00, 1996-06-27 09:00:00}|2.0               |
|5     |{1997-03-13 09:00:00, 1997-03-20 09:00:00}|3.7735849056603774|
|6     |{1999-12-09 09:00:00, 1999-12-16 09:00:00}|4.153846153846154 |
|7     |{1996-06-20 09:00:00, 1996-06-27 09:00:00}|3.64              |
|8     |{1998-03-19 09:00:00, 1998-03-26 09:00:00}|3.625             |
|8     |{1998-03-26 09:00:00, 1998-04-02 09:00:00}|3.0               |
|9     |{1997-03-20 09:00:00, 1997-03-27 09:00:00}|3.865168539325843 |
|10    |{2008-11-20 09:00:00, 2008-11-27 09:00:00}|3.452830188679245 |
|11    |{2008-04-10 09:00:00, 2008-04-17 09:00:00}|3.1458333333333335|
+------+------------------------------------------+------------------+
only showing top 20 rows
```
