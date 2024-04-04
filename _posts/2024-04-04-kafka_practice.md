---
title: Apache Kafka 실습
date: 2024-04-04 12:00:00 +/-09:00
categories: [데이터파이프라인]
author: yehoon
tags: [Kafka, Docker, DevOps, Python]
description: 
---

## docker-compose.yml 파일 생성
```yml
version: '3'
services:
  zookeeper:
    image: wurstmeister/zookeeper:latest
    ports:
      - "2181:2181"
  kafka:
    image: wurstmeister/kafka:latest
    ports:
      - "9092:9092"
    environment:
      KAFKA_ADVERTISED_LISTENERS: INSIDE://kafka:9093,OUTSIDE://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT
      KAFKA_LISTENERS: INSIDE://0.0.0.0:9093,OUTSIDE://0.0.0.0:9092
      KAFKA_INTER_BROKER_LISTENER_NAME: INSIDE
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
```

## docker compose 실행
```bash
docker-compose up -d
```
`kafka_practice-kafka-1`와 `kafka_practice-zookeeper-1` 컨테이너 생성

## topic 생성
`tasks` topic 생성
```bash
docker exec -it kafka_practice-kafka-1 /opt/kafka/bin/kafka-topics.sh --create --topic tasks --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
```

## topic 조회
```bash
docker exec kafka_practice-kafka-1 /opt/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

## producer 생성
```bash
docker exec -it kafka_practice-kafka-1 /opt/kafka/bin/kafka-console-producer.sh --topic tasks --bootstrap-server localhost:9092
```

## consumer 생성
```bash
docker exec -it kafka_practice-kafka-1 /opt/kafka/bin/kafka-console-consumer.sh --topic tasks --bootstrap-server localhost:9092 --from-beginning
```

## topic 제거
```bash
docker exec kafka_practice-kafka-1 /opt/kafka/bin/kafka-topics.sh --delete --topic tasks --bootstrap-server localhost:9092
```

## Python으로 실행
### 라이브러리 설치
```bash
pip install kafka-python
```

### Producer
```python
from kafka import KafkaProducer
import json

# Configure Kafka connection parameters
bootstrap_servers = 'localhost:9092'  # Replace with Kafka container IP
topic = 'tasks'

# Create Kafka producer instance
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Produce messages
for i in range(10):
    message = {'message': f'Message {i}'}
    producer.send(topic, value=message)
    print(f"Produced: {message}")

# Close the producer
producer.close()
```

### Consumer
```python
from kafka import KafkaConsumer
import json

# Configure Kafka connection parameters
bootstrap_servers = 'localhost:9092'  # Replace with Kafka container IP
topic = 'tasks'
consumer_group_id = 'my_consumer_group'

# Create Kafka consumer instance
consumer = KafkaConsumer(topic,
                         group_id=consumer_group_id,
                         bootstrap_servers=bootstrap_servers,
                         value_deserializer=lambda x: json.loads(x.decode('utf-8')))

# Consume messages
for message in consumer:
    print(f"Consumed: {message.value}")

# Close the consumer
consumer.close()
```

{% include embed/youtube.html id='3hIn8x58U1E' %}
