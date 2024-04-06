---
title: Django 기반 웹 개발(2)
date: 2024-04-06 11:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django]
description: Django 기반 웹 개발
---

전체 레시피 조회, 레시피 검색, 레시피 태그을 구현할 때 진행 상태를 확인하려면 데이터가 필요하다.  
그래서 더미 데이터를 생성하기로 했다.  
그리고 GraphQL을 활용하여 사용자 데이터 생성까지 해보았다.

# 개발 2일차
## 1. 레시피 데이터 생성
1. LLM을 활용하여 `json` 형식의 [레시피 데이터](https://github.com/yehoon17/recipe_management_system/blob/master/data/recipes_fraction.json) 생성  

2. [fraction_to_decimal.py](https://github.com/yehoon17/recipe_management_system/blob/master/data/fraction_to_decimal.py)를 사용하여 데이터를 분수 형식에서 [소수 형식](https://github.com/yehoon17/recipe_management_system/blob/master/data/recipes.json)으로 변환

3. [edit_json.py](https://github.com/yehoon17/recipe_management_system/blob/master/data/edit_json.py)를 사용하여 데이터 안에 [불필요한 key 제거](https://github.com/yehoon17/recipe_management_system/blob/master/data/updated_recipes.json)

4. [get_recipe_images.py](https://github.com/yehoon17/recipe_management_system/blob/master/data/get_recipe_images.py)를 사용하여 이미지 파일 웹 크롤링

5. 이미지 등록 안된 레시피에 default 이미지 [적용](https://github.com/yehoon17/recipe_management_system/commit/fac4556c41b84bfb2a2bdf69f587470658d59a88)

## 2. 사용자 데이터 생성
1. 사용자 데이터 생성하는 [GraphQL mutation](https://github.com/yehoon17/recipe_management_system/commit/e2035047f55e9ff3f769947a6251fef07e79220f) 구현
2. 사용자 데이터 생성하는 [GraphQL API 요청](https://github.com/yehoon17/recipe_management_system/commit/92651bc5e4372c61c9d5148ba82bce59c96fe2fa) 구현



## 3. 레시피 편집 개선
레시피 생성할 때, 재료와 태그가 이미 존재하더라도 새로 생성하는 것을 확인했다.  
그래서 재료와 태그의 속성을 중복 불가하게 수정했다.
그리고 레시피 생성할 때 재료와 태그가 이미 존재하면 존재하는 데이터를 가져오게 구현했다.   

[수정한 내용](https://github.com/yehoon17/recipe_management_system/commit/0483f6b6015dd31c90564d0983cd8dfbcc5964f2)
   

# 후기
[2일차 코드](https://github.com/yehoon17/recipe_management_system/tree/e976d63ad7bd29bbef95149306d6c210c56d61f2)

레시피 데이터 생성를 생성하는 데 llama API와 function calling을 적용해보려고 했다.  
그러나 결과의 퀄리티가 별로여서 ChatGPT로 생성한 데이터로 사용했다. 
바로 ChatGPT로 생성한 데이터를 사용했다면 2일차 안에 핵심 기능은 전부 구현했을 것이다.
그래도 function calling에 대해 알아보는 [유익한 경험](https://github.com/yehoon17/recipe_management_system/blob/llama_api/data_generation/llama_api/api_test.ipynb)을 했다.


# TODO
 - [ ] 전체 레시피 조회
 - [ ] 레시피 검색
 - [ ] 레시피 태그
 - [x] 데이터 생성
 - [x] 로그인 안 되어있을 때, 레시피 생성 버튼 클릭시 로그인 화면으로 전환
 - [x] 이미지 등록 안된 레시피에 default 이미지 적용
 - [ ] 스타일 

---

Next: [Django 기반 웹 개발(3)](https://yehoon17.github.io/posts/django_web_dev_3/)



