---
title: SQL 정리
date: 2024-05-10 12:00:00 +09:00
categories: [SQL]
author: yehoon
tags: [SQL, OracleDB, MariaDB]
---

## SQL
**1. SELECT 문:**
   ```
   SELECT column1, column2 FROM table_name WHERE condition;
   ```
   - 지정된 열에서 조건에 따라 데이터를 검색

**2. WHERE 절:**
   ```
   SELECT * FROM table_name WHERE condition;
   ```
   - 지정된 조건에 따라 테이블의 행을 필터링

**3. AND/OR 연산자:**
   ```
   SELECT * FROM table_name WHERE condition1 AND condition2;
   SELECT * FROM table_name WHERE condition1 OR condition2;
   ```
   - 여러 조건을 결합하여 행을 필터링하는 데 사용
   - `AND`는 두 조건이 모두 `TRUE`여야 함
   - `OR`은 적어도 하나의 조건이 `TRUE`여야 함

**4. ORDER BY 절:**
   ```
   SELECT * FROM table_name ORDER BY column_name ASC|DESC;
   ```
   - 지정된 열을 오름차순(`ASC`) 또는 내림차순(`DESC`)으로 정렬

여러 열에 대한 정렬
```
SELECT column1, column2, ...
FROM table_name
ORDER BY column1 ASC|DESC, column2 ASC|DESC, ...;
```

**5. DISTINCT 키워드:**
   ```
   SELECT DISTINCT column_name FROM table_name;
   ```
   - 지정된 열의 고유한 값을 반환

**6. INSERT INTO 문:**
   ```
   INSERT INTO table_name (column1, column2) VALUES (value1, value2);
   ```
   - 지정된 열 값으로 지정된 테이블에 새로운 데이터 행을 추가

**7. UPDATE 문:**
   ```
   UPDATE table_name SET column1 = value1, column2 = value2 WHERE condition;
   ```
   - 지정된 조건에 따라 기존 데이터를 수정

**8. DELETE 문:**
   ```
   DELETE FROM table_name WHERE condition;
   ```
   - 지정된 조건에 따라 테이블에서 하나 이상의 행을 제거

**9. CREATE TABLE 문:**
   ```
   CREATE TABLE table_name (
       column1 datatype,
       column2 datatype,
       ...
   );
   ```
   - 지정된 열과 데이터 유형으로 새로운 테이블을 생성

**10. ALTER TABLE 문:**
   ```
   ALTER TABLE table_name ADD column_name datatype;
   ```
   - 기존 테이블에 새 열을 추가

**11. DROP TABLE 문:**
   ```
   DROP TABLE table_name;
   ```
   - 전체 테이블과 그 데이터를 삭제

**12. JOIN 절:**
   ```
   SELECT * FROM table1 INNER JOIN table2 ON table1.column = table2.column;
   ```
   - 관련 열을 기준으로 여러 테이블에서 데이터를 검색

**13. GROUP BY 절:**
   ```
   SELECT column1, COUNT(column2) FROM table_name GROUP BY column1;
   ```
   - 지정된 열에서 동일한 값을 가진 행을 그룹화하고 `COUNT`와 같은 집계 함수를 사용

**14. HAVING 절:**
   ```
   SELECT column1, COUNT(column2) FROM table_name GROUP BY column1 HAVING condition;
   ```
   - 지정된 조건에 따라 그룹화된 행을 필터링

**15. LIKE 연산자:**
   ```
   SELECT * FROM table_name WHERE column_name LIKE pattern;
   ```
   - 지정된 열에서 지정된 패턴에 따라 행을 필터링
   - Wildcard
     - `%`: 0 또는 여러 문자
     - `_`: 단일 문자

**16. IN 연산자:**
   ```
   SELECT * FROM table_name WHERE column_name IN (value1, value2, ...);
   ```
   - 목록의 값 중 하나와 일치하는지 확인

<ins>⚠ OracleDB에서는 `IN` 연산자의 목록이 1000개로 제한됨</ins>  
해결 방법:  
1. 쿼리 분할

    ```
    SELECT * FROM your_table WHERE your_column IN (value1, value2, ..., value1000)
    UNION ALL
    SELECT * FROM your_table WHERE your_column IN (value1001, value1002, ..., value2000)
    UNION ALL
    ...
    ```

2. 서브퀴리 또는 임시 테이블 사용
    ```
    SELECT * FROM your_table WHERE your_column IN (SELECT value FROM your_values_table)
    ```

3. 테이블 또는 common table expression (CTE) 조인
    ```
    WITH values_cte AS (
        SELECT value FROM your_values_table
    )
    SELECT t.* FROM your_table t
    JOIN values_cte v ON t.your_column = v.value;
    ```

4. 비교값 변환
    ```
    SELECT * FROM your_table WHERE (1, your_column) IN ((1, value1), (1, value2), ...)
    ```


**17. BETWEEN 연산자:**
   ```
   SELECT * FROM table_name WHERE column_name BETWEEN value1 AND value2;
   ```
   - 값 범위에 따라 행을 필터링
   - `value1`, `value2` 포함

**18. NULL 값:**
   ```
   SELECT * FROM table_name WHERE column_name IS NULL;
   ```
   - 지정된 열에서 `NULL` 값을 가진 행을 필터링

**19. LIMIT 절 (MySQL 및 PostgreSQL용):**
   ```
   SELECT * FROM table_name LIMIT number;
   ```
   - 결과 집합에서 반환되는 행 수를 제한

**20. OFFSET 절 (MySQL 및 PostgreSQL용):**
   ```
   SELECT * FROM table_name LIMIT number OFFSET offset;
   ```
   - 결과 집합을 반환하기 전에 지정된 행 수를 건너뜀


\* Oracle SQL에서는 `ROWNUM` 또는 `FETCH FIRST n ROWS ONLY` 사용
```
SELECT * 
FROM your_table
WHERE ROWNUM <= number;
```

```
SELECT * 
FROM your_table
FETCH FIRST number ROWS ONLY;
```




**21. CASE 문:**
   ```
   SELECT column_name,
          CASE 
              WHEN condition1 THEN 'Result1'
              WHEN condition2 THEN 'Result2'
              ELSE 'Default Result'
          END AS new_column_name
   FROM table_name;
   ```
   - `SELECT` 문 내에서 조건부 논리를 수행
   - 각 조건을 순서대로 평가하고 `TRUE` 인 첫 번째 조건의 결과를 반환


**22. EXISTS 연산자:**
   ```
   SELECT column_name FROM table_name WHERE EXISTS (SELECT * FROM other_table WHERE condition);
   ```
   - 하위 쿼리의 행의 존재 여부를 확인하고 하위 쿼리가 적어도 한 행을 반환하면 `TRUE` 를 반환
   - 그렇지 않으면 FALSE를 반환

**23. UNION 연산자:**
   ```
   SELECT column1 FROM table1
   UNION
   SELECT column1 FROM table2;
   ```
   - 두 개 이상의 `SELECT` 문의 결과 집합을 하나의 결과 집합으로 결합하고 기본적으로 중복 행을 제거

**24. WITH 절 - Common Table Expressions(CTEs):**
   ```
   WITH cte_name AS (
       SELECT column1, column2 FROM table_name
   )
   SELECT * FROM cte_name WHERE condition;
   ```
   - 동일한 쿼리 내에서 참조할 수 있는 임시 결과 집합인 Common Table Expressions(CTE)을 정의
