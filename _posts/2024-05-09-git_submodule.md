---
title: Git Submodule
date: 2024-05-09 09:00:00 +09:00
categories: [Git]
author: yehoon
tags: [Version Control, Submodule]
---

## Git Submodule이란?
 - 다른 Git 저장소를 작업 Git 저장소 안에 서브디렉터리로 포함할 수 있게 해주는 기능
   - 주로 프로젝트에 외부 종속성이나 라이브러리를 포함하고 싶을 때 사용
   - 메인 프로젝트 저장소와 외부 저장소를 분리

## 1. 서브모듈 추가
1. 프로젝트 내에서 서브모듈을 추가하고 싶은 위치 설정
2. 추가하려는 저장소의 URL 복사
3. `git submodule add <url> <추가할 위치>` 명령어 사용
- 서브모듈과 `.gitmodules` 생성됨
  - 이미 다른 서브모듈이 추가된 상태라면 `.gitmodules`에 내용이 추가됨
- `git status`로 서브모듈과 `.gitmodules`가 staged된 걸 확인할 수 있음

- Git은 서브모듈을 인식하며 해당 디렉터리에 있지 않을 때 그 내용을 추적하지 않는다
  - 대신, Git은 해당 저장소의 특정 커밋으로 본다
  - `git diff  --cached`로 확인

## 2. 서브모듈 초기화 및 업데이트
 - 서브모듈을 추가한 후에는 반드시 초기화하고 업데이트해야함
 - `git submodule update --init` 명령은 서브모듈 저장소를 프로젝트 디렉터리로 복제하고 적절한 커밋을 확인
    - 프로젝트가 서브모듈의 상태와 동기화

## 3. 서브모듈 사용
- 서브모듈이 추가되고 초기화되면, 해당 서브모듈을 다른 Git 저장소와 마찬가지로 사용할 수 있다
  - 변경사항을 만들고 커밋하고 저장소에 푸시할 수 있다
  - 서브모듈을 최신 커밋으로 업데이트하려면 서브모듈 디렉터리로 이동하여 변경 사항을 가져와야 한다
```bash
cd <서브모듈 위치>
git pull origin main
```

## 4. 서브모듈이 있는 저장소 복제
 - 서브모듈이 있는 저장소를 복제할 때는 서브모듈도 초기화 및 업데이트해야함
```bash
git clone --recursive <repository-url>
```
 - 또는 이미 저장소를 복제한 경우:
```bash
git submodule update --init
```


참고: <https://git-scm.com/book/en/v2/Git-Tools-Submodules>
