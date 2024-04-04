---
title: Apache Kafka 입문
date: 2024-04-04 12:00:00 +/-09:00
categories: [데이터파이프라인]
author: yehoon
tags: [Kafka, DevOps]
description: 
---

대량의 queries per second(QPS)를 처리하기 위해, 비동기 작업 관리 및 폭발적인 요청 처리 기술이 필요하다고 한다. 그리고 Kafka 같은 Message Broker가 그 중 하나라고 한다.

일단 개념을 이해하기 위해 영상을 찾아봤다.


[System Design: Apache Kafka In 3 Minutes](https://www.youtube.com/watch?v=HZklgPkboro)
 - Kafka는 데이터 생성자(Producer)로부터 데이터를 전달 받아, 데이터 소비자(Consumer)한테 제공한다.
 - 실시간 스트리밍, 대량 프래픽에 유리하다.


[System Design: Why is Kafka fast?](https://www.youtube.com/watch?v=UNUz1-msbOM)
 - 대량의 데이터를 효율적으로 전달하는데 유리하다.
 - 빠르게 하는 디자인
    1. Sequencial I/O
        Random 접근이 아닌 Sequencial 접근으로 읽고 쓰기 빠르게 함  
        Append Only

    2. 데이터 이동 전략
        Read with zero copy
        Direct Memory Access(DMA) 활용


[3. Apache Kafka Fundamentals | Apache Kafka Fundamentals](https://www.youtube.com/watch?v=B5j3uNBH8X4)
 - 구성
   - Producer: 데이터 생성자
   - Kafka Brokers: 데이터 전달자
   - Consumer: 데이터 소비자
   - ZooKeeper: Kafka Brokers 관리
 - Decoupling Producers and Consumers
   - 모두 서로 신경 안 쓰고 독립적
 - ZooKeeper 역할
   - Broker가 죽으면 어떻게 처리하는지 관리 등
 - Topic, Partition, Segment
   - Topic: 데이터 주제 단위
   - Partition: Topic을 나눈거
   - Segment: Partition을 나눈거
 - Log란
   - "Immutable records of things and write to a log at the very end"  
 - Data Elements
   - Header, Key, Value, timestamp




Next: [Apache Kafka 실습](https://yehoon17.github.io/posts/kafka_practice/)
