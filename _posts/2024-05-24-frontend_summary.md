---
title: Frontend 정리
date: 2024-05-24 12:00:00 +09:00
categories: [Frontend]
author: yehoon
tags: [html, css, javascript]
---

## html
### 1. 기본 구조와 문법

#### 기본 구조:
- **`<!DOCTYPE html>`**
  - 문서 유형 및 HTML 버전을 선언
  - 브라우저가 페이지를 올바르게 렌더링하도록 도와줌
- **`<html>`**
  - HTML 문서의 루트 요소
- **`<head>`**
  - 문서에 대한 메타 정보를 포함
  - 예) 제목, 문자 세트, 연결된 스타일시트 및 스크립트
- **`<body>`**
  - 사용자에게 표시되는 HTML 문서의 내용을 포함

```html
<!DOCTYPE html>
<html>
<head>
    <title>페이지 제목</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <!-- 본문 -->
</body>
</html>
```

### 2. 일반 HTML 태그

#### 텍스트 포맷:
- **`<p>`**: 단락
- **`<h1>` ~ `<h6>`**: 제목
- **`<strong>`**: <strong>중요한 텍스트</strong>
- **`<em>`**: <em>강조된 텍스트</em>
- **`<small>`**: <small>작은 텍스트</small>
- **`<mark>`**: <mark>강조된 텍스트</mark>


#### 링크와 이미지:
- **`<a>`**: 하이퍼링크
- **`<img>`**: 이미지

```html
<a href="https://www.example.com">링크</a>
<img src="image.jpg" alt="이미지 설명">
```

#### 목록:
- **`<ul>`**: 정렬되지 않은 목록(unordered list)
- **`<ol>`**: 순서가 있는 목록(ordered list)
- **`<li>`**: 목록 항목

```html
<ul>
    <li>항목 1</li>
    <li>항목 2</li>
</ul>
<ol>
    <li>항목 1</li>
    <li>항목 2</li>
</ol>
```

### 3. 양식과 입력 요소

- **`<form>`**: 사용자 입력을 위한 HTML 양식
- **`<input>`**: 입력 컨트롤
- **`<textarea>`**: 여러 줄의 텍스트 입력 컨트롤
- **`<select>`**: 드롭다운 목록

일반 입력 유형:
- **`type="text"`**: 한 줄 텍스트 입력
- **`type="password"`**: 비밀번호 입력(글자가 숨겨짐)
- **`type="email"`**: 이메일 주소 입력
- **`type="number"`**: 숫자 입력
- **`type="date"`**: 날짜 입력
- **`type="checkbox"`**: 체크박스 입력
- **`type="radio"`**: 라디오 버튼 입력
- **`type="submit"`**: 제출 버튼

```html
<form action="/submit-form" method="post">
    <label for="fname">이름:</label>
    <input type="text" id="fname" name="fname">
    <label for="lname">성:</label>
    <input type="text" id="lname" name="lname">
    <input type="submit" value="제출">
</form>
```

### 4. 시맨틱 HTML

**시맨틱 HTML 요소**: 웹 콘텐츠에 의미를 부여
- **`<header>`**: 문서 또는 섹션의 헤더
- **`<nav>`**: 탐색 링크
- **`<main>`**: 문서의 주요 콘텐츠
- **`<section>`**: 문서의 섹션을 그룹화
- **`<article>`**: 독립적이고 독립된 콘텐츠
- **`<footer>`**: 문서 또는 섹션의 footer
- **`<aside>`**: 주요 콘텐츠 이외의 콘텐츠


### 5. 속성 및 전역 속성

**속성**: 요소에 대한 추가 정보를 제공
- **일반 속성**: `id`, `class`, `style`, `title`, `href`, `src`, `alt`, `width`, `height`.

**전역 속성**:
- **`id`**: 요소의 고유 식별자
- **`class`**: CSS 및 JavaScript에 사용되는 클래스 이름
- **`style`**: in-line CSS 스타일
- **`title`**: 요소에 대한 추가 정보를 제공
- **`data-*`**: 사용자 정의 데이터 속성
- **`contenteditable`**: 요소의 내용이 편집 가능한지 여부
- **`hidden`**: 요소를 숨김
- **`tabindex`**: 요소의 탭 순서


### 6. 문서 객체 모델 (DOM)
HTML 문서의 프로그래밍 인터페이스
- 객체 트리로 문서 구조를 나타냄
  - **요소**: HTML 요소에 해당하는 DOM 트리의 노드
  - **JavaScript 상호 작용**: JavaScript는 이러한 요소를 선택하고 조작할 수 있음

## CSS

### 1. 기본 선택자
   - **요소 선택자** `element {}`  
      태그 이름을 기반으로 요소를 선택 
   - **클래스 선택자** `.class {}`   
     클래스 속성을 기반으로 요소를 선택
   - **아이디 선택자** `#id {}`  
     고유한 아이디 속성을 기반으로 단일 요소를 선택
   - **속성 선택자** `[type="text"] {}`  
    속성 및 속성 값에 따라 요소를 선택
   - **Universal 선택자** `* {}`  
    HTML 문서의 모든 요소를 선택
   - **하위 선택자** `ancestor descendant {}`  
    다른 요소의 하위 요소를 선택

### 2. 박스 모델
   - **Content**: 텍스트 또는 이미지와 같은 실제 내용
   - **Padding**: 컨텐츠와 테두리 사이의 공간
   - **Border**: 패딩을 둘러싼 테두리
   - **Margin**: 테두리 바깥의 공간으로, 요소를 다른 요소와 분리
  
  ![alt text](/assets/img/frontend/box_model.png)


### 3. 디스플레이 속성
   - **블록** `display: block;`  
     가능한 전체 너비를 차지하고 수직으로 쌓이는 요소
   - **인라인** `display: inline;`  
     필요한 만큼의 너비를 차지하고 새로운 줄에서 시작하지 않는 요소
   - **인라인-블록** `display: inline-block;`  
     인라인 요소처럼 동작하지만 너비와 높이를 설정할 수 있음
   - **플렉스박스** `display: flex;`  
     컨테이너 내 요소의 유연하고 효율적인 정렬 및 공간 분배를 가능하게 하는 레이아웃 모델
     - 플렉스 방향: `flex-direction`
     - 정렬 내용: `justify-content`
     - 항목 정렬: `align-items`
     - 플렉스 확장: `flex-grow`
     - 플렉스 축소: `flex-shrink`
     - 플렉스 기초: `flex-basis`
   - **그리드** `display: grid;`  
     페이지를 행과 열로 나누어 요소를 정렬하는 레이아웃 모델
     - 그리드 템플릿 열: `grid-template-columns`
     - 그리드 템플릿 행: `grid-template-rows`
     - 그리드 열 간격: `grid-column-gap`
     - 그리드 행 간격: `grid-row-gap`
     - 그리드 열: `grid-column`
     - 그리드 행: `grid-row`

### 4. 포지셔닝
   - **정적** `position: static;`  
     요소가 문서의 정상적인 흐름에 따라 배치
   - **상대적** `position: relative;`  
     요소가 자신의 정상적인 위치를 기준으로 배치
   - **절대적** `position: absolute;`  
     요소가 가장 가까운 포지셔닝된 조상을 기준으로 배치
   - **고정적**  `position: fixed;`  
     요소가 브라우저 창을 기준으로 배치되며 페이지를 스크롤할 때 움직이지 않음
   - **붙박이** `position: sticky;`  
     요소가 사용자의 스크롤 위치를 기준으로 배치되며 지정된 지점에 도달하면 고정 위치로 전환

### 5. 텍스트 스타일링
- **색상**: `color`
- **글꼴 크기**: `font-size`
- **글꼴 패밀리**: `font-family`
- **글꼴 무게**: `font-weight`
- **텍스트 장식**: `text-decoration`
- **텍스트 정렬**: `text-align`
- **라인 높이**: `line-height`
- **글자 간격**: `letter-spacing`

### 6. 색상과 배경
- **배경 색상**: `background-color`
- **배경 이미지**: `background-image`
- **배경 위치**: `background-position`
- **배경 반복**: `background-repeat`
- **배경 크기**: `background-size`

### 7. 변형 & 전이
- **변형**: `transform`
- **전이**: `transition`
- **전이 속성**: `transition-property`
- **전이 기간**: `transition-duration`
- **전이 타이밍 함수**: `transition-timing-function`
- **전이 지연**: `transition-delay`

### 8. 기타
- **불투명도**: `opacity`
- **박스 그림자**: `box-shadow`
- **텍스트 그림자**: `text-shadow`
- **오버플로우**: `overflow`
- **커서**: `cursor`
- **Z-인덱스**: `z-index`

## JavaScript

### 1. 변수와 데이터 타입
```javascript
// 변수
let name = "John";
const age = 30;
var count = 0;

// 데이터 타입
let str = "Hello"; // 문자열
let num = 10; // 숫자
let bool = true; // 불리언
let arr = [1, 2, 3]; // 배열
let obj = { name: "John", age: 30 }; // 객체
```

### 2. 함수
```javascript
// 함수 선언
function greet(name) {
    return "안녕하세요, " + name + "님!";
}

// 화살표 함수
const greet = (name) => {
    return `안녕하세요, ${name}님!`;
};

// 함수 호출
console.log(greet("John")); // 출력: 안녕하세요, John님!
```

### 3. DOM 조작
```javascript
// ID로 요소 가져오기
const element = document.getElementById("myElement");

// 클래스 이름으로 요소 가져오기
const elements = document.getElementsByClassName("myClass");

// 이벤트 리스너 추가
element.addEventListener("click", () => {
    console.log("클릭!");
});

// 요소 스타일 변경
element.style.color = "red";

// 요소 내용 수정
element.innerHTML = "새로운 내용";
```

### 4. 조건문
```javascript
// If 문
if (age >= 18) {
    console.log("어른");
} else {
    console.log("아이");
}

// 삼항 연산자
const message = age >= 18 ? "어른" : "아이";
```

### 5. 반복문
```javascript
// For 루프
for (let i = 0; i < 5; i++) {
    console.log(i);
}

// While 루프
let i = 0;
while (i < 5) {
    console.log(i);
    i++;
}
```

### 6. 배열
```javascript
// 배열 생성
const fruits = ["사과", "바나나", "오렌지"];

// 요소 접근
console.log(fruits[0]); // 출력: 사과

// 요소 추가
fruits.push("포도");

// 요소 제거
fruits.pop();
```

### 7. 객체
```javascript
// 객체 생성
const person = {
    name: "John",
    age: 30,
    address: {
        city: "뉴욕",
        country: "미국"
    }
};

// 속성 접근
console.log(person.name); // 출력: John

// 속성 수정
person.age = 35;
```

### 8. Promise
JavaScript에서 비동기 작업을 처리하기 위한 객체
 - 주로 네트워크 요청, 파일 읽기, 타임아웃 설정 등의 비동기 작업을 다룰 때 사용
 - 성공 또는 실패 상태를 가질 수 있으며, 이 상태에 따라 처리할 콜백 함수를 등록할 수 있음

```javascript
// 프로미스 생성
const fetchData = () => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve("데이터를 성공적으로 가져옴");
        }, 2000);
    });
};

// 프로미스 사용
fetchData()
    .then(data => console.log(data))
    .catch(error => console.error("데이터 가져오기 오류:", error));
```

### 9. Fetch API
```javascript
// API에서 데이터 가져오기
fetch("https://api.example.com/data")
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error("데이터 가져오기 오류:", error));
```

- `fetch()`: 주어진 URL에서 네트워크 요청을 생성하고, 해당 URL에 대한 응답을 Promise로 반환
- `.then()`: Promise가 이행될 때 실행할 콜백 함수를 정의
  - 첫 번째 `.then()`
    - 응답 객체를 매개변수로 받아서 응답을 JSON 형식으로 파싱
    - `response.json()`은 응답 본문을 JSON 형식으로 변환하는 Promise를 반환
  - 두 번째 `.then()`
    - 이전 Promise가 이행되면 실행
    -  파싱된 JSON 데이터가 매개변수로 전달되며, 여기서는 간단히 콘솔에 데이터를 출력
-  `.catch()`: Promise가 거부될 때 실행할 콜백 함수를 정의
   -  네트워크 요청이 실패하거나 응답을 받지 못할 때 발생하는 오류를 처리
   -  여기서는 오류 메시지를 콘솔에 출력

### 10. ES6 기능
```javascript
// 해체 할당
const { name, age } = person;

// 전개 연산자
const numbers = [1, 2, 3];
const newNumbers = [...numbers, 4, 5];

// 템플릿 리터럴
const message = `이름: ${name}, 나이: ${age}`;
```

