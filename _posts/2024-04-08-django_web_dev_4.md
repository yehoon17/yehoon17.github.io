---
title: Django 기반 웹 개발(4)
date: 2024-04-08 19:25:00 +09:00
categories: [프로젝트, 웹 개발]
author: yehoon
tags: [Django, Backend]
description: Django 기반 웹 개발
---

레시피를 수정할 때, 이미 등록된 재료를 수정하거나 삭제하는 기능 구현

### 재료 수정
```python 
ingredient_names = request.POST.getlist('ingredient_name[]')
ingredient_quantities = request.POST.getlist('ingredient_quantity[]')
ingredient_units = request.POST.getlist('ingredient_unit[]')

for name, quantity, unit in zip(ingredient_names, ingredient_quantities, ingredient_units):
    ingredient, created = Ingredient.objects.get_or_create(name=name)
    defaults = {
        'quantity': quantity,
        'unit': unit
    }
    RecipeIngredient.objects.update_or_create(
        recipe=recipe,
        ingredient=ingredient,
        defaults=defaults
    )
```

`RecipeIngredient.objects.update_or_create()`에서 `recipe`와 `ingredient` 기준으로 데이터를 확인하여 이미 데이터베이스에 존재하면 `default`로 업데이터하고 존재하지 않으면 새로 생성한다.

[update_or_create 문서](https://docs.djangoproject.com/en/5.0/ref/models/querysets/#update-or-create)

### 재료 삭제
```python
ingredient_names = request.POST.getlist('ingredient_name[]')
ingredient_quantities = request.POST.getlist('ingredient_quantity[]')
ingredient_units = request.POST.getlist('ingredient_unit[]')

ingredients = []
for name, quantity, unit in zip(ingredient_names, ingredient_quantities, ingredient_units):
    # 재료 수정 생략
    ingredients.append(ingredient)

recipe_ingredient_to_delete = RecipeIngredient.objects.filter(
    recipe=recipe
).exclude(
    Q(ingredient__in=ingredients)
)
recipe_ingredient_to_delete.delete()
```

[exclude 문서](https://docs.djangoproject.com/en/5.0/ref/models/querysets/#exclude)  
[Q 문서](https://docs.djangoproject.com/en/5.0/ref/models/querysets/#q-objects)

---

Next: [Django 기반 웹 개발(5)](https://yehoon17.github.io/posts/django_web_dev_5/)



