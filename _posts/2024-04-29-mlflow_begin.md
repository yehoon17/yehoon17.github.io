---
title: MLflow 입문
date: 2024-04-29 21:00:00 +09:00
categories: [Machine Learning]
author: yehoon
tags: [MLflow, MLOps]
image: /assets/img/mlflow/thumbnail.png

---

## MLflow란?
- 완전한 머신 러닝 라이프사이클을 관리하는 플랫폼
- 실험 추적, 재현 가능한 모델 개발, 모델 관리 및 배포를 통합하는 데 사용됨
  - 팀원 간의 협업을 용이하게 하고, 모델 관리와 버전 관리를 효율적으로 처리함
  - 빠른 모델 반복을 가능케 하여 더 빠르게 최적의 모델을 발견하고 개선할 수 있음
  - 머신 러닝 프로젝트를 확장 가능하게 만들어 복잡한 문제에 대응할 수 있음
  - 실험 결과와 모델 학습 과정을 재현할 수 있어 일관된 결과를 도출할 수 있음
  - 모델을 쉽게 배포하고 관리할 수 있어, 제품에 적용하기에 준비를 완료할 수 있음

## 주요 구성 요소:

1. **추적(Tracking)**:
   - 데이터 과학자가 여러 실행을 구성하고 비교할 수 있는 도구를 제공하여 매개변수, 메트릭 및 출력 파일을 포함하여 실험을 추적하는 데 도움
   - 이를 통해 모델 개발 중에 쉽게 실험하고 반복 가능

2. **프로젝트(Projects)**:
   - 코드, 종속성 및 환경 사양을 재현 가능한 형식으로 패키징
   - 이를 통해 한 환경에서 개발된 모델을 다른 환경에서 쉽게 재현하고 배포 가능

3. **모델(Models)**:
   - 머신 러닝 모델을 패키징하기 위한 표준 형식을 제공하여 모델을 쉽게 배포
   - 이러한 모델은 다양한 배포 도구와 프레임워크와 원활하게 통합되어 개발 환경에서 제품 환경으로의 전환을 원활하게 함

4. **레지스트리(Registry)**:
   - 머신 러닝 모델을 관리하고 버전을 관리하는 중앙 집중식 저장소 역할
   - 팀원 간의 협업을 용이하게 하고 모델이 일관되게 추적, 유효성 검사 및 배포되도록 보장

5. **배포(Deployment)**:
   - 클라우드 서비스, Kubernetes 및 엣지 장치를 포함한 다양한 플랫폼에 모델을 배포할 수 있음
   - 이 유연성을 통해 조직은 클라우드나 엣지와 같은 환경에 모델을 배포 가능

6. **통합(Integration)**:
   - TensorFlow, PyTorch, scikit-learn 및 Spark MLlib와 같은 인기있는 머신 러닝 라이브러리 및 프레임워크와 원활하게 통합됨
   - 이를 통해 데이터 과학자는 기존 도구와 워크플로우를 활용하면서 MLflow의 기능을 활용할 수 있음

## 튜토리얼
### 설치 
```bash
pip install mlflow
```

### 로컬에 MLflow 실행
```bash
mlflow ui
```

 - 명령어 실행한 위치에 `mlruns`와 `mlartifacts` 생성됨
 - `http://localhost:5000` 에 접속

### 예시 코드 
<https://mlflow.org/docs/latest/getting-started/intro-quickstart/notebooks/tracking_quickstart.html>

1. notebook 다운로드
2. uri 수정
   ```python 
   mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
   ```
3. 코드 실행 및 확인

![alt text](/assets/img/mlflow/overview.png){: width="500" height="400" }
_Overview_

![alt text](/assets/img/mlflow/artifacts.png){: width="500" height="400" }
_Artifacts_
