---
title: "TypeError: SQLAlchemy.create_all() got an unexpected keyword argument 'app'"
date: 2024-04-08 23:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Flask]
description: Flask 버전 업그레이드 에러
---

2년 전에 기획했던 웹 프로젝트를 확인해 보았다.  
Flask 및 기타 패키지의 버전을 업데이트하고 실행해봤는데 이런 에러가 발생했다.  

```
TypeError: SQLAlchemy.create_all() got an unexpected keyword argument 'app'
```

구글링하니 바로 [관련 글](https://stackoverflow.com/questions/73968584/flask-sqlalchemy-db-create-all-got-an-unexpected-keyword-argument-app)을 찾을 수 있었고, 바로 해결할 수 있었다. 

> Flask-SQLAlchemy 3는 이제 create_all과 같은 메소드에 app 인자를 받지 않습니다. 대신 항상 활성화된 Flask 애플리케이션 컨텍스트를 요구합니다. 그리고 create_database 함수는 필요하지 않습니다. SQLAlchemy는 이미 존재하는 파일을 덮어쓰지 않으며, 데이터베이스가 생성되지 않는 유일한 경우는 오류를 발생시켰을 때입니다.
{: .prompt-info }

차차 [이 프로젝트](https://github.com/yehoon17/PoE-Gem-Flipping)도 개발할 계획이다.
