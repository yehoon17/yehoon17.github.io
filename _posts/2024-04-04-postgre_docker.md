---
title: Docker로 PostgreSQL 실행하기
date: 2024-04-04 09:00:00 +/-09:00
categories: [데이터베이스 , 데이터베이스 관리]
author: yehoon
tags: [PostgreSQL, Docker, DevOps]
render_with_liquid: false
description: Docker로 PostgreSQL 실행하기
---

## 1. Docker 설치
Docker 설치가 안 되어 있다면, Docker는 공식 웹사이트를 참고하여 설치

## 2. PostgreSQL 이미지 가져오기
Docker 이미지: 컨테이너를 생성하기 위한 설계도  
Docker Hub: 다양한 이미지를 제공하는 공식 저장소     

Docker Hub에서 공식 PostgreSQL 이미지를 가져오기  
```bash
docker pull postgres
```

## 3. Docker 컨테이너 생성
Docker 컨테이너: Docker 이미지를 기반으로 실행되는 가상화된 환경  
두가지 방법이 있다
 - PostgreSQL 이미지에서 바로 생성
 - Dockerfile을 통해서 생성

### 3.1. PostgreSQL 이미지에서 바로 생성
```bash
docker run --name postgres_db -e POSTGRES_PASSWORD=mysecretpassword -d postgres
```


### 3.2. Dockerfile을 통해서 생성
#### Dockerfile 생성
```Dockerfile
# Use the official PostgreSQL image from Docker Hub
FROM postgres:latest

# Set environment variables
ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword
ENV POSTGRES_DB=mydatabase

# Copy initialization scripts to be executed when container starts
COPY init.sql /docker-entrypoint-initdb.d/

# Expose PostgreSQL port
EXPOSE 5432
```
#### init.sql 생성
```sql
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE
);
```

#### Docker 이미지 빌드 
```bash
docker build -t my-postgres-image .
```

#### Docker 컨테이너 실행
```bash
docker run --name postgres_db -d -p 5432:5432 my-postgres-image
```

## 4. 컨테이너 상태 확인
```bash
docker ps
```

## 5. 컨테이너 내부에 들어가기
**PostgreSQL 이미지에서 바로 생성했을 경우**
```bash
docker exec -it postgres_db psql -U postgres
```

**Dockerfile을 통해서 생성**
```bash
docker exec -it postgres_db psql -U myuser mydatabase
```


## 6. PostgreSQL 사용
**PostgreSQL 이미지에서 바로 생성했을 경우**
```sql
CREATE DATABASE mydatabase;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE
);
```

### 데이터 입력
```sql
INSERT INTO users (username, email) VALUES 
('user1', 'user1@example.com'),
('user2', 'user2@example.com'),
('user3', 'user3@example.com');
```



## 7. 컨테이너 중지 및 삭제
컨테이너 중지
```bash
docker stop postgres_db
```

컨테이너 삭제
```bash
docker rm postgres_db
```
