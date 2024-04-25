---
title: Django 기반 웹 개발(6)
date: 2024-04-25 13:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django, Backend]
description: Django 기반 웹 개발
---

댓글 기능 구현

작업 브랜치: <https://github.com/yehoon17/recipe_management_system/tree/comment>

## 테스트
```python 
from django.test import TestCase
from django.urls import reverse
from .models import Recipe, Comment, User


class CommentCRUDTests(TestCase):
    def setUp(self):
        # 테스트 유저 생성
        self.user = User.objects.create_user(username='testuser', password='12345')

        # 테스트 레시피 생성
        self.recipe = Recipe.objects.create(
            user=self.user,
            title='Test Recipe',
            description='Test Description',
            preparation_time=30,
            cooking_time=45,
            difficulty_level='Easy'
        )

    def test_create_comment_not_logged_in(self):
        comment_text = "This is a test comment"
        response = self.client.post(reverse('create_comment', args=[self.recipe.id]), {
            'text': comment_text
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertRedirects(response, reverse('login') + '?next=' + reverse('create_comment', args=[self.recipe.id]))

    def test_create_comment_logged_in(self):
        self.client.login(username='testuser', password='12345')
        
        comment_text = "This is a test comment"
        response = self.client.post(reverse('create_comment', args=[self.recipe.id]), {
            'text': comment_text
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(Comment.objects.filter(text=comment_text).exists())
        self.assertRedirects(response, reverse('recipe_detail', args=[self.recipe.id]))

    def test_read_comment(self):
        n_comments = 10
        comments = []
        for i in range(n_comments):
            comment = Comment.objects.create(
                user=self.user,
                recipe=self.recipe,
                text=f"Test Comment {i}"
            )
            comments.append(comment)
        
        response = self.client.get(reverse('recipe_detail', args=[self.recipe.id]))

        self.assertEqual(response.status_code, 200)
        for comment in comments:
            self.assertContains(response, comment.text)

    def test_update_comment(self):
        self.client.login(username='testuser', password='12345')

        comment = Comment.objects.create(
            user=self.user,
            recipe=self.recipe,
            text="Original Comment"
        )

        updated_text = "Updated Comment"
        response = self.client.post(reverse('update_comment', args=[self.recipe.id, comment.id]), {
            'text': updated_text
        })

        self.assertEqual(response.status_code, 302)  
        self.assertTrue(Comment.objects.filter(text=updated_text).exists())

    def test_delete_comment(self):
        self.client.login(username='testuser', password='12345')

        comment = Comment.objects.create(
            user=self.user,
            recipe=self.recipe,
            text="To be deleted"
        )

        response = self.client.post(reverse('delete_comment', args=[self.recipe.id, comment.id]))

        self.assertEqual(response.status_code, 302)  
        self.assertFalse(Comment.objects.filter(text="To be deleted").exists())


    def test_reply_comment(self):
        self.client.login(username='testuser', password='12345')

        parent_comment = Comment.objects.create(
            user=self.user,
            recipe=self.recipe,
            text="Parent Comment"
        )

        comment_text = "Child Comment"
        response = self.client.post(reverse('reply_comment', args=[self.recipe.id, parent_comment.id]), {
            'text': comment_text
        }, follow=True)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(Comment.objects.filter(text=comment_text, parent_comment=parent_comment).exists())
```

## 구현
![alt text](/assets/img/recipe/comment.png)


---

Next: [Django 기반 웹 개발(6)](https://yehoon17.github.io/posts/django_web_dev_7/)



