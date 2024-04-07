---
title: REST API와 GraphQL API 비교
date: 2024-04-07 14:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django, GraphQL, REST API]
description: REST API와 GraphQL API 비교
---

[Django 프로젝트](https://github.com/yehoon17/recipe_management_system)에서 REST API를 구현하고, GraphQL과 비교해봤다.

작업 브랜치: [restapi](https://github.com/yehoon17/recipe_management_system/tree/restapi)

## REST API 구현
1. `Recipe`, `Ingredient`, `RecipeIngredient에` 대한 Serializer 생성
   

```python
from rest_framework import serializers
from .models import Recipe, Ingredient, RecipeIngredient 


class RecipeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recipe
        fields = [
            'id',
            'user',
            'title',
            'description',
            'preparation_time',
            'cooking_time',
            'difficulty_level'
        ]


class IngredientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ingredient
        fields = [
            'id',
            'name'
        ]


class RecipeIngredientSerializer(serializers.ModelSerializer):
    recipe = RecipeSerializer
    Ingredient =IngredientSerializer

    class Meta:
        model = RecipeIngredient
        fields = [
            'recipe',
            'ingredient',
            'quantity',
            'unit'
        ]
```

2. `views.py`에 ListCreateAPIView 생성
   
```python
from rest_framework import generics
from .serializers import RecipeSerializer, IngredientSerializer, RecipeIngredientSerializer


class RecipeListCreate(generics.ListCreateAPIView):
    queryset = Recipe.objects.all()
    serializer_class = RecipeSerializer

class IngredientListCreate(generics.ListCreateAPIView):
    queryset = Ingredient.objects.all()
    serializer_class = IngredientSerializer

class RecipeIngredientListCreate(generics.ListCreateAPIView):
    queryset = RecipeIngredient.objects.all()
    serializer_class = RecipeIngredientSerializer

```

3. `urls.py` 설정
   

```python
    path('rest-api/recipe/', views.RecipeListCreate.as_view()),
    path('rest-api/ingredient/', views.IngredientListCreate.as_view()),
    path('rest-api/recipe-ingredient/', views.RecipeIngredientListCreate.as_view())
```

## REST API와 GraphQL API 비교
### 환경 설정
```python
import requests
import pandas as pd
```

### Over fetching

모든 레시피의 id, title, difficultyLevel를 조회하는 경우를 살펴보자.

#### REST API


```python
recipe_list_url = 'http://localhost:8000/rest-api/recipe/'

response = requests.get(recipe_list_url)
```
    


```python
recipe_list = response.json()
df = pd.DataFrame(recipe_list)
df.head()
```




|   id |   user | title              | description                       |   preparation_time |   cooking_time | difficulty_level   |
|-----:|-------:|:-------------------|:----------------------------------|-------------------:|---------------:|:-------------------|
|    1 |      2 | Korean Bibimbap zz | Bibimbap is a signature Korean... |                 30 |             25 | Moderate           |
|    2 |      2 | Kimchijeon         | Kimchijeon is a savory Korean ... |                 15 |             15 | Easy               |
|    3 |      2 | Dubu Jorim         | Dubu Jorim is a flavorful Kore... |                 15 |             20 | Easy               |
|    4 |      3 | 계란 후라이        | egg fry                           |                  0 |              3 | 쉬움               |
|    5 |      2 | Scramble Egg       | Egg                               |                  0 |              3 | Easy               |



레시피의 user, description, preparation_time, cooking_time까지 조회되게 된다.

#### GraphQL




```python
recipe_query = """query MyQuery {
  allRecipes {
    id
    title
    difficultyLevel
  }
}"""
```


```python
graphql_url = 'http://localhost:8000/graphql/'  

response = requests.get(graphql_url, json={'query': recipe_query})
```


```python
recipe_list = response.json()['data']['allRecipes']
df = pd.DataFrame(recipe_list)
df.head()
```




|   id | title              | difficultyLevel   |
|-----:|:-------------------|:------------------|
|    1 | Korean Bibimbap zz | Moderate          |
|    2 | Kimchijeon         | Easy              |
|    3 | Dubu Jorim         | Easy              |
|    4 | 계란 후라이        | 쉬움              |
|    5 | Scramble Egg       | Easy              |



### Under fetching

모든 레시피의 id, title, difficultyLevel와 레시피에 들어가는 재료의 정보를 조회하는 경우를 살펴보자.


#### REST API



```python
recipe_list_url = 'http://localhost:8000/rest-api/recipe/'
response = requests.get(recipe_list_url)

recipe_list = response.json()
recipe_df = pd.DataFrame(recipe_list)
recipe_df.head()
```


|   id |   user | title              | description                       |   preparation_time |   cooking_time | difficulty_level   |
|-----:|-------:|:-------------------|:----------------------------------|-------------------:|---------------:|:-------------------|
|    1 |      2 | Korean Bibimbap zz | Bibimbap is a signature Korean... |                 30 |             25 | Moderate           |
|    2 |      2 | Kimchijeon         | Kimchijeon is a savory Korean ... |                 15 |             15 | Easy               |
|    3 |      2 | Dubu Jorim         | Dubu Jorim is a flavorful Kore... |                 15 |             20 | Easy               |
|    4 |      3 | 계란 후라이        | egg fry                           |                  0 |              3 | 쉬움               |
|    5 |      2 | Scramble Egg       | Egg                               |                  0 |              3 | Easy               |


```python
recipe_ingredient_list_url = 'http://localhost:8000/rest-api/recipe-ingredient/'
response = requests.get(recipe_ingredient_list_url)

recipe_ingredient_list = response.json()
recipe_ingredient_df = pd.DataFrame(recipe_ingredient_list)
recipe_ingredient_df.head()
```



|   recipe |   ingredient |   quantity | unit        |
|---------:|-------------:|-----------:|:------------|
|        1 |            1 |          4 | cups        |
|        1 |            2 |        200 | grams       |
|        1 |            3 |          3 | tablespoons |
|        1 |            4 |          2 | tablespoons |
|        1 |            5 |          1 | tablespoons |


```python
ingredient_list_url = 'http://localhost:8000/rest-api/ingredient/'
response = requests.get(ingredient_list_url)
ingredient_list = response.json()
ingredient_df = pd.DataFrame(ingredient_list)
ingredient_df.head()
```


|   id | name        |
|-----:|:------------|
|    1 | Cooked rice |
|    2 | Beef        |
|    3 | Soy sauce   |
|    4 | Sesame oil  |
|    5 | Sugar       |



```python
cols = ['id', 'title', 'difficulty_level']
recipe_df = recipe_df[cols]
```


```python
df = (
    recipe_ingredient_df
    .merge(recipe_df, left_on='recipe', right_on='id', how='left')
    .drop(columns=['id'])
    .merge(ingredient_df, left_on='ingredient', right_on='id', how='left')
    .drop(columns=['id', 'ingredient'])
    .rename(columns={'name': 'ingredient_name', 'recipe': 'id'})
)
df.head()
```



|   id |   quantity | unit        | title              | difficulty_level   | ingredient_name   |
|-----:|-----------:|:------------|:-------------------|:-------------------|:------------------|
|    1 |          4 | cups        | Korean Bibimbap zz | Moderate           | Cooked rice       |
|    1 |        200 | grams       | Korean Bibimbap zz | Moderate           | Beef              |
|    1 |          3 | tablespoons | Korean Bibimbap zz | Moderate           | Soy sauce         |
|    1 |          2 | tablespoons | Korean Bibimbap zz | Moderate           | Sesame oil        |
|    1 |          1 | tablespoons | Korean Bibimbap zz | Moderate           | Sugar             |




필요한 정보를 전부 가져오기 위해 여러 번의 요청을 보내야 한다. 

#### GraphQL




```python
recipe_ingredient_query = """query MyQuery {
  allRecipes {
    id
    title
    difficultyLevel
    recipeingredientSet {
      ingredient {
        name
      }
      quantity
      unit
    }
  }
}"""
```


```python
response = requests.get(graphql_url, json={'query': recipe_ingredient_query})
recipe_list = response.json()['data']['allRecipes']
```



```python
df = pd.json_normalize(
    recipe_list,
    record_path='recipeingredientSet',
    meta=['id', 'title', 'difficultyLevel']
)
df.head()
```



|   quantity | unit        | ingredient.name   |   id | title              | difficultyLevel   |
|-----------:|:------------|:------------------|-----:|:-------------------|:------------------|
|          4 | cups        | Cooked rice       |    1 | Korean Bibimbap zz | Moderate          |
|        200 | grams       | Beef              |    1 | Korean Bibimbap zz | Moderate          |
|          3 | tablespoons | Soy sauce         |    1 | Korean Bibimbap zz | Moderate          |
|          2 | tablespoons | Sesame oil        |    1 | Korean Bibimbap zz | Moderate          |
|          1 | tablespoons | Sugar             |    1 | Korean Bibimbap zz | Moderate          |


한번의 요청으로 필요한 데이터를 전부 가져온다.