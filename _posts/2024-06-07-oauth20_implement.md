---
title: OAuth 2.0 구현
date: 2024-06-07 13:00:00 +09:00
categories: [Backend]
author: yehoon
tags: [Authentication, Django, Flask]
image: /assets/img/oauth/thumbnail.png
---

1. Django 기반 웹을 Resource Owner로서 OAuth 2.0 기능 구현  
2. OAuth 2.0을 Flask 앱에 Client로서 구현  

## 1. Django 기반 웹에 OAuth 2.0 기능 구현
### 1.1. Django OAuth Toolkit 설치
```bash
pip install django-oauth-toolkit
```

### 1.2. INSTALLED_APPS에 oauth2_provider 추가
Django 프로젝트의 `settings.py` 파일에서 `oauth2_provider`를 `INSTALLED_APPS` 목록에 추가
```python
INSTALLED_APPS = [
      ...
      'oauth2_provider',
]
```

### 1.3. oauth2_provider URL 추가
`urls.py`
```python
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('oauth/', include('oauth2_provider.urls', namespace='oauth2_provider')),
]
```

### 1.4. 마이그레이션 실행
OAuth 2.0에 필요한 데이터베이스 테이블을 생성하기 위해 마이그레이션을 실행

```bash
python manage.py migrate
```

### 1.5. 서버 실행
```bash
python manage.py runserver
```

### 1.6. OAuth 2.0 애플리케이션 생성
1. Django Admin 이동하여 Application 추가
![](/assets/img/oauth/add_oauth_app.png)
2. Application 설정
   - Client Type: Confidential
   - Authorization Grant Type: Authorization Code
   - Redirect URIs: `http://localhost:5000/callback`

> Redirect URI: 승인 후에 어디로 가는지

3. Client id와 Client secret 복사해두기 


## 2. OAuth 2.0을 Flask 앱에 Client로서 구현
### 2.1. 라이브러리 설치
```bash
pip install flask flask_session flask_sqlalchemy pkce
```

### 2.2. Flask 앱 구현
```python
from flask import Flask, redirect, url_for, session, request, jsonify
import requests
import pkce
from flask_session import Session
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure Flask-Session
app.config['SESSION_TYPE'] = 'sqlalchemy'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sessions.db'
db = SQLAlchemy(app)
app.config['SESSION_SQLALCHEMY'] = db
Session(app)

CLIENT_ID = 'ihfF8TmV2ggMuUgsbDEpDvSGvKG2xMT4T3qAhmHS'
CLIENT_SECRET = 'Pko6fQYlEIcg3CcLRe1mqqF2X5ZbcKBMExan42iAmQmJKpri1T35dYfoNbIHZo2UsdTAN0aL6BxIMvHgOGCRDv1jg2XXYBjA8n2jj4GEiLaMFBhx2w3kpyAWfo3AvMl3'
AUTHORIZATION_BASE_URL = 'http://localhost:8000/oauth/authorize/'
TOKEN_URL = 'http://localhost:8000/oauth/token/'
REDIRECT_URI = 'http://localhost:5000/callback'

@app.route('/')
def home():
    return '''
        <h1>Flask OAuth Client</h1>
        <a href="/login">Login with OAuth</a>
    '''

@app.route('/login')
def login():
    code_verifier = pkce.generate_code_verifier(length=128)
    session['code_verifier'] = code_verifier
    code_challenge = pkce.get_code_challenge(code_verifier)

    authorization_url = f"{AUTHORIZATION_BASE_URL}?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT_URI}&code_challenge={code_challenge}&code_challenge_method=S256"
    return redirect(authorization_url)

@app.route('/callback')
def callback():
    print(session)
    error = request.args.get('error')
    if error:
        return f"Error: {error}"

    code = request.args.get('code')
    if not code:
        return 'Missing code parameter.'

    code_verifier = session.get('code_verifier')
    if not code_verifier:
        return 'Missing code verifier in session.'

    token_response = requests.post(TOKEN_URL, data={
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': REDIRECT_URI,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'code_verifier': code_verifier,
    })
    token_json = token_response.json()

    if 'error' in token_json:
        return f"Error: {token_json['error']} - {token_json.get('error_description')}"

    session['oauth_token'] = token_json['access_token']
    return redirect(url_for('profile'))

@app.route('/profile')
def profile():
    token = session.get('oauth_token')
    if token is None:
        return redirect(url_for('login'))
    
    response = requests.get('http://localhost:8000/protected/', headers={
        'Authorization': f'Bearer {token}'
    })
    return jsonify(response.json())


if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### 2.3. Flask 앱 실행
```bash
python app.py
```

## 3. 결과
<https://github.com/yehoon17/recipe_management_system/tree/oauth>
### 3.1. Flask 앱 접속
`http://127.0.0.1:5000/` 
![](/assets/img/oauth/flask_login.png)

### 3.2. 로그인 
1. Django 웹사이트의 로그인 페이지로 이동하게 됨
   ![](/assets/img/oauth/login.png)

2. 로그인하면 권한 요청 페이지로 이동함
   ![](/assets/img/oauth/authorize.png)

3. Flask 앱이 Access Token을 요청

4. Django 웹사이트에서 Access Token을 발급 

5. Access Token을 사용하여 Django의 view 접근
   