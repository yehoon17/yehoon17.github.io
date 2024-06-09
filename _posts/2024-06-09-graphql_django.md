---
title: Django에 GraphQL 구현하기
date: 2024-06-09 10:00:00 +09:00
categories: [Backend]
author: yehoon
tags: [API, Django]
image: /assets/img/graphql/thumbnail.png
---


## Django 프로젝트에 GraphQL 설정하기
### 1. 라이브러리 설치
   ```bash
   pip install graphene-django
   ```

### 2. 설치된 앱에 Graphene-Django 추가
   `settings.py` 수정
   ```python
   INSTALLED_APPS = [
       ...
       'graphene_django',
   ]
   
   GRAPHENE = {
       'SCHEMA': 'yourapp.schema.schema',  # 스키마가 정의된 곳
   }
   ```

### 3. GraphQL 스키마 작성
   `schema.py` 파일에 GraphQL 스키마 정의

   1. Django 모델 정의
   2. DjangoObjectType 정의 [(참고)](https://docs.graphene-python.org/projects/django/en/latest/queries/)
      - 필드 설정: `fields`, `exclude`
   3. 쿼리 정의 [(참고)](https://docs.graphene-python.org/projects/django/en/latest/queries/)
   4. 뮤테이션 정의 [(참고)](https://docs.graphene-python.org/projects/django/en/latest/mutations/)
   5. 스키마 정의
  ```python
  schema = graphene.Schema(query=Query, mutation=Mutation)
  ```
   

### 4. URL 구성
   ```python
   # urls.py
   from django.urls import path
   from graphene_django.views import GraphQLView
   from django.views.decorators.csrf import csrf_exempt

   urlpatterns = [
       path("graphql/", csrf_exempt(GraphQLView.as_view(graphiql=True))),
   ]
   ```
   - `GraphQLView`는 GraphQL API의 엔드포인트를 제공
   - `graphiql=True`는 GraphQL을 탐색하고 테스트할 수 있는 웹 기반 IDE인 GraphiQL 인터페이스를 활성화함


## 예시
[schema.py](https://github.com/yehoon17/recipe_management_system/blob/master/website/recipes/schema.py)
   
### 기본 쿼리
```python 
class RecipeType(DjangoObjectType):
  class Meta:
      model = Recipe

class Query(graphene.ObjectType):
    all_recipes = graphene.List(RecipeType)

    def resolve_all_recipes(self, info, **kwargs):
        return Recipe.objects.all()
```

![](/assets/img/graphql/recipe_query.png)
_등록된 모든 레시피 조회_

### 조건 쿼리
```python
class UserType(DjangoObjectType):
    class Meta:
        model = User
        
class Query(graphene.ObjectType):
    all_users = graphene.List(UserType, is_superuser=graphene.Boolean())

    def resolve_all_users(self, info, is_superuser=None):
        queryset = User.objects.all()
        if is_superuser is not None:
            queryset = queryset.filter(is_superuser=is_superuser)
        return queryset
```

![](/assets/img/graphql/user_query.png)
_superuser가 아닌 유저 조회_

```python
class Query(graphene.ObjectType):
    recipes_by_title = graphene.List(RecipeType, title=graphene.String())

    def resolve_recipes_by_title(self, info, title):
        return Recipe.objects.filter(title__icontains=title)
```
   
![](/assets/img/graphql/recipe_title_query.png)
_제목에 "chicken" 포함된 레시피 조회_


### 뮤테이션

#### 생성
```python
class CreateUser(graphene.Mutation):
    class Arguments:
        username = graphene.String(required=True)
        email = graphene.String(required=True)
        password = graphene.String(required=True)

    user = graphene.Field(UserType)

    def mutate(self, info, username, email, password):
        user = User.objects.create_user(username=username, email=email, password=password)
        return CreateUser(user=user)

class Mutation(graphene.ObjectType):
    create_user = CreateUser.Field()
```

![](/assets/img/graphql/create_user.png)
_유저 생성_

#### 수정

```python
class UpdateRecipe(graphene.Mutation):
    class Arguments:
        id = graphene.Int(required=True)
        new_title = graphene.String()

    recipe = graphene.Field(RecipeType)

    def mutate(self, info, id, new_title):
        recipe = Recipe.objects.get(id=id)
        recipe.title = new_title
        recipe.save()
        return UpdateRecipe(recipe=recipe)
    

class Mutation(graphene.ObjectType):
    update_recipe = UpdateRecipe.Field()
```

![](/assets/img/graphql/recipe_update_before.png)
_제목 변경 전_

![](/assets/img/graphql/recipe_update_after.png)
_제목 변경 후_
