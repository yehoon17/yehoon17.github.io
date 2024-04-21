---
title: Django 기반 웹 개발(3)
date: 2024-04-07 10:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django, GraphQL, Backend]
description: Django 기반 웹 개발
---

레시피 데이터를 준비했으니까 데이터베이스에 입력하고 나머지를 구현했다.

## 개발 3일차

### 1. 레시피 데이터 입력
1. [레시피 데이터 json 파일](https://github.com/yehoon17/recipe_management_system/blob/master/data/recipes.json)을 그대로 사용하기 귀찮게 중간에 불필요한 key가 존재했다. 
    [edit_json.py](https://github.com/yehoon17/recipe_management_system/blob/master/data/edit_json.py)를 사용하여 데이터 안에 [불필요한 key 제거](https://github.com/yehoon17/recipe_management_system/blob/master/data/updated_recipes.json)했다.

2. 데이터베이스에 입력하기
   Django command로 실행시킬 수 있게 구현해봤다.
   [load_recipe_data.py](https://github.com/yehoon17/recipe_management_system/blob/master/website/recipes/management/commands/load_recipe_data.py)

### 2. 레시피 전체 조회 페이지 
[Merge branch 'recipe_list'](https://github.com/yehoon17/recipe_management_system/commit/326030af926f1664bfe5c386daf2d9d75bc22324)
![recipe](assets/img/recipe/list_all.png)

### 3. 태그로 레시피 조회
[Implemented feature to display all recipes associated with a clicked tag](https://github.com/yehoon17/recipe_management_system/commit/5e487d87e0d33789b3d5d4ce35932f9f9d5e6e73)

### 4. 제목으로 레시피 검색
[Implement searching recipe with recipe's title](https://github.com/yehoon17/recipe_management_system/commit/4736bf2a21252ee011492138fb6e57e3c807846c)

### 5. Rating 구현
1. 레시피의 Rating 평균 계산하여 display
2. 로그인되어 있다면, Rating 입력
3. 로그인 안 되어 있다면, Rating 입력 시도할 경우 로그인 화면으로 redirect  
   
[Merge pull request #3 from yehoon17/rating](https://github.com/yehoon17/recipe_management_system/commit/2b3e9c264d4bcce259ba4b41eb12183979b63a54)

### 6. GraphQL 구현
1. Query
    - 레시피 제목으로 레시피 조회
    - `cooking_time`으로 정렬하여 레시피 조회
    - 재료를 포함하는 레시피 조회

2. Mutation
    - 레시피 생성
    - 레시피 수정

[Merge pull request #4 from yehoon17/graphql](https://github.com/yehoon17/recipe_management_system/commit/ca208b8bee6db0b4b25a7aa2adb3d07b32eeea1f)
   
### 7. 스타일링
작업 브랜치: [style](https://github.com/yehoon17/recipe_management_system/tree/style)   
HTML/CSS로 스타일 적용

### 8. Docker containerize
1. [Dockerfile](https://github.com/yehoon17/recipe_management_system/blob/master/Dockerfile) 생성
2. 테스트

## 결과
{% include embed/youtube.html id='G7DPu1fAI1Q' %}


## 후기
[3일차 코드](https://github.com/yehoon17/recipe_management_system/tree/5970c0a5b864b94e27218685bb1bc20244364420)

일단 핵심 기능들은 구현하고, 스타일링도 했다. 추후에 개선하고 더 추가할 생각이다. 
 - [ ] 댓글 구현
 - [ ] 재료 삭제 구현
 - [ ] 재료 수정 구현
 - [ ] 태그 삭제 구현
 - [ ] 태그 수정 구현
 - [ ] rating 중복 불가 구현
 - [ ] 로그 구현
 - [ ] 테스트 구현
 - [ ] 개인 프로필
 - [ ] 작성한 레시피 조회
 - [ ] WEB, WAS, 라우팅 구현
 - [ ] REST API 구현
 - [ ] REST API와 GraphQL API 비교해보기 


---

Next: [Django 기반 웹 개발(4)](https://yehoon17.github.io/posts/django_web_dev_4/)



