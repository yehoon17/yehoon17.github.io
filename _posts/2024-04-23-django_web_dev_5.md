---
title: Django 기반 웹 개발(5)
date: 2024-04-23 03:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django, Backend]
description: Django 기반 웹 개발
---

프로필 조회 및 수정, 작성한 레시피 조회 기능 구현

작업 브랜치: <https://github.com/yehoon17/recipe_management_system/tree/profile>

## 프로필 조회 및 작성한 레시피 조회
![alt text](/assets/img/recipe/profile.png)

### views.py
```python
@login_required
def profile(request):
    recipes = Recipe.objects.filter(user=request.user)
    context = {'recipes': recipes}
    return render(request, 'profile/profile.html', context)
```

## 프로필 수정
![alt text](/assets/img/recipe/profile_edit.png)

### forms.py
```python
class ProfileEditForm(UserChangeForm):
    email = forms.EmailField()
    first_name = forms.CharField(max_length=30)
    last_name = forms.CharField(max_length=30)

    class Meta:
        model = User
        fields = ['email', 'first_name', 'last_name']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fields['password']

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
        return user
    
class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['image']
```

### views.py
```python
@login_required
def profile_edit(request):
    if request.method == 'POST':
        # Handle profile image form
        profile_image_form = ProfileForm(request.POST, request.FILES, instance=request.user.profile)
        if profile_image_form.is_valid():
            profile_image_form.save()
        
        # Handle user information form
        user_form = ProfileEditForm(request.POST, instance=request.user)
        if user_form.is_valid():
            user_form.save()

        return redirect('profile')  # Redirect to the profile page after editing
    else:
        profile_image_form = ProfileForm(instance=request.user.profile)
        user_form = ProfileEditForm(instance=request.user)
    return render(request, 'profile/profile_edit.html', {'profile_image_form': profile_image_form, 'user_form': user_form})
```

## Thoughts
 - 작성한 레시피 내에서 검색 기능
 - 작성한 레시피의 태그
 - 작성한 레시피 통계
  
---

Next: [Django 기반 웹 개발(6)](https://yehoon17.github.io/posts/django_web_dev_6/)



