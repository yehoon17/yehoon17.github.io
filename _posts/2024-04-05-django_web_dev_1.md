---
title: Django 기반 웹 개발(1)
date: 2024-04-05 14:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django, GraphQL, CRUD, Backend]
description: Django 기반 웹 개발
---

단기간 안에 Django 기반 웹을 개발해보기로 했다.  
주제는 요리 레시피 블로그로 정했다.  

## 개발 1일차
### 1. 프로젝트 기획
#### 개요
Django와 GraphQL로 구축된 레시피 관리 시스템  
사용자는 레시피를 생성, 조회, 편집 및 삭제와 레시피 검색 가능 

#### 기능
- 사용자 인증: 사용자는 회원가입, 로그인 및 로그아웃하여 자신의 레시피를 관리할 수 있습니다.
- 레시피 CRUD 작업: 사용자는 레시피를 생성, 조회, 편집 및 삭제할 수 있습니다.
- 태깅: 레시피에 카테고리를 추가하여 조직 및 필터링이 쉽도록 합니다.
- 평가 시스템: 사용자는 1에서 5까지의 별점으로 레시피를 평가할 수 있습니다.
- 검색 기능: 사용자는 제목, 재료 또는 태그를 기준으로 레시피를 검색할 수 있습니다.

#### ER 다이어그램
![erd](https://github.com/yehoon17/recipe_management_system/blob/master/document/er_diagram.png?raw=true)

### 2. 개발
#### 2.1. 환경 설정
**git 생성**
```bash
git init
```

**가상 환경 생성**
```bash
python -m venv venv
```

**Django 프로젝트 생성**
```bash
django-admin startproject website
```

**Django app 생성**
```bash
python manage.py startapp recipes
```


#### 2.2. 구현
1. Django 모델 생성
2. URL 설정
3. HTML 생성
4. CSS 생성
5. Header, Footer 생성
6. 로그인, 로그아웃 기능 구현
7. 레시피 CRUD 기능 구현
   1. 로그인 되었을 때, 레시피 생성 가능
   2. 로그인된 계정이 작성한 레시피만 편집 및 제거 가능
   3. 테이블 형식에 재료 입력, 행 추가 및 제거 가능
8. [GraphQL 시작](https://github.com/yehoon17/recipe_management_system/commit/420cdf9f9b9a4e4b4f2f11e76e4e19609b313c18)
   1. 모든 레시피 조회, 모든 재료 조회하는 Query 생성
   2. URL 설정
   3. `settings.py` 설정


#### 2.3. 에러 해결 
1. 커스텀 User 사용 에러 [해결](https://github.com/yehoon17/recipe_management_system/commit/5f1f20042db595bff933d9f14baa1140b4540ab0)  
   
> auth.User.groups: (fields.E304) Reverse accessor 'Group.user_set' for 'auth.User.groups' clashes with reverse accessor for 'recipes.User.groups'.
{: .prompt-warning }

### 3. 결과
![1일차 홈페이지](assets/img/recipe/homepage_1.png)
_홈페이지_

![create recipe](assets/img/recipe/create_1.png)
_레시피 생성_


![ingredient_table_1](assets/img/recipe/ingredient_table_1.png)
_재료 입력_


![recipe detail](assets/img/recipe/detail_1.png)
_레시피 조회_

[1일차 코드](https://github.com/yehoon17/recipe_management_system/tree/d7bd211731c0bb64fe13912535995f9b79e69177)


### 4. TODO
 - [ ] 전체 레시피 조회
 - [ ] 레시피 검색
 - [ ] 레시피 태그
 - [ ] 로그인 안 되어있을 때, 레시피 생성 버튼 클릭시 로그인 화면으로 전환
 - [ ] 이미지 등록 안된 레시피에 default 이미지 적용
 - [ ] 스타일 

---

Next: [Django 기반 웹 개발(2)](https://yehoon17.github.io/posts/django_web_dev_2/)



