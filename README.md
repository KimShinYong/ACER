# ACER (Actor-Critic with Experience Replay) - CartPole 구현

## 1. 프로젝트 개요

이 프로젝트는 `CartPole-v1` 환경에서 **ACER(Actor-Critic with Experience Replay)** 알고리즘을 학습하는 예제입니다.


이 구현은 ACER의 핵심 아이디어를 이해하고 실행해볼 수 있도록 단순화한 교육용 구현입니다.

---

## 2. ACER란?

ACER는 **Actor-Critic with Experience Replay**의 약자입니다.

A2C/A3C와 같은 Actor-Critic 계열 알고리즘은 정책 기반 강화학습의 장점을 가지지만, 일반적으로 **현재 정책으로 수집한 데이터만 사용하는 on-policy 방식**에 가깝기 때문에 샘플 효율이 낮습니다.

반면 ACER는 과거 경험을 저장한 뒤 다시 꺼내 쓰는 **Experience Replay**를 도입하여 데이터 효율을 높입니다.  
하지만 과거 정책으로 수집된 데이터를 현재 정책 학습에 그대로 사용하면 학습이 불안정해질 수 있기 때문에, 다음과 같은 보정 기법을 사용합니다.

- Importance Sampling
- Truncated Importance Sampling
- Bias Correction
- Retrace

즉, ACER는 **A3C의 안정성**과 **Replay Buffer 기반 데이터 재사용**을 함께 가져가려는 알고리즘입니다.

---

## 3. 주요 특징

이 코드의 주요 특징은 다음과 같습니다.

### 3.1 Actor-Critic 구조
하나의 신경망에서 정책과 Q값을 함께 출력합니다.

- `pi(x)` : 상태에서 각 행동의 확률 출력
- `q(x)` : 상태에서 각 행동의 Q값 출력
- `v(x)` : `sum(pi * q)` 방식으로 상태가치 계산

### 3.2 병렬 환경 사용
`ParallelEnv` 클래스를 통해 여러 개의 CartPole 환경을 동시에 실행합니다.  
이 구조는 A2C/A3C 스타일 코드와 유사하게 구성되어 있습니다.

### 3.3 Experience Replay
수집한 rollout을 `ReplayBuffer`에 저장한 뒤, 이후 다시 샘플링하여 off-policy 학습에 사용합니다.

### 3.4 Off-policy Correction
과거 정책(`mu`)과 현재 정책(`pi`)의 차이를 Importance Sampling으로 보정합니다.

### 3.5 Truncated Importance Sampling
importance ratio가 너무 커질 경우 분산이 커질 수 있으므로 일정 값으로 잘라서 사용합니다.

### 3.6 Bias Correction
importance ratio를 잘라내면서 발생할 수 있는 편향을 줄이기 위해 보정항을 추가합니다.

### 3.7 Retrace 스타일 Q-return
뒤에서부터 거꾸로 return을 계산하며, off-policy 학습에서도 비교적 안정적인 target을 만들도록 구성했습니다.

---

## 4. 실행 환경
### 4.1 Python 버전

다음과 같은 Python 환경을 권장합니다.

Python 3.10 이상

### 4.2 필요 라이브러리

아래 라이브러리가 필요합니다.

torch

gymnasium

numpy

설치 명령어:
```
pip install torch gymnasium numpy
```

---

# 5. 코드 구성 설명
### 5.1 ActorCritic

정책과 Q함수를 동시에 추정하는 신경망입니다.

주요 메서드

forward(x) : 공통 feature를 통과시켜 policy logits와 Q값을 반환

pi(x) : softmax를 적용한 정책 확률 반환

q(x) : 각 행동에 대한 Q값 반환

v(x) : 정책확률과 Q값을 이용해 상태가치 계산

### 5.2 ParallelEnv

여러 개의 환경을 병렬로 실행하기 위한 클래스입니다.

역할

여러 CartPole 환경을 동시에 step 수행

rollout 수집 속도 향상

A2C/A3C 스타일의 병렬 환경 구조 유지

주요 메서드

reset() : 모든 환경 초기화

step(actions) : 각 환경에 행동 적용

close() : 모든 환경 종료

### 5.3 ReplayBuffer

수집한 rollout을 저장하는 버퍼입니다.

역할

on-policy rollout 저장

이후 off-policy replay update에 사용

주요 메서드

put(rollout) : rollout 저장

sample(n) : 저장된 rollout 중 일부 샘플링

size() : 현재 저장 개수 반환

### 5.4 make_rollout

현재 정책으로 일정 길이의 trajectory를 수집합니다.

저장 정보

상태 s

행동 a

보상 r

종료 여부 마스크 mask

행동 당시 정책 분포 mu

마지막 상태 s_last

여기서 mu를 저장하는 이유는, ACER가 과거 정책과 현재 정책의 차이를 비교해 importance ratio를 계산하기 때문입니다.

### 5.5 acer_update

ACER의 핵심 학습 함수입니다.

수행 내용

rollout 데이터를 텐서로 변환

마지막 상태로부터 초기 q_ret 계산

뒤에서부터 Retrace 스타일로 target 갱신

importance ratio 계산

truncated importance sampling 적용

bias correction 적용

actor loss, critic loss 계산

entropy 정규화 추가

역전파 및 최적화 수행

---

# 6. 학습 로직 요약

전체 학습 흐름은 다음과 같습니다.

병렬 환경에서 rollout 수집

수집한 rollout을 replay buffer에 저장

현재 rollout으로 on-policy update 수행

replay buffer에서 과거 rollout을 샘플링

샘플링한 rollout으로 off-policy replay update 수행

일정 주기마다 테스트 진행

---

# 7. 하이퍼파라미터 설명

코드 상단의 주요 하이퍼파라미터는 다음 의미를 가집니다.

n_train_processes : 병렬 환경 개수

learning_rate : Adam optimizer 학습률

gamma : 할인율

update_interval : 한 번 rollout에서 수집할 step 수

max_train_steps : 전체 학습 step 수

print_interval : 로그 출력 주기

buffer_limit : replay buffer 최대 저장 개수

replay_ratio : rollout 하나당 replay update 수행 횟수

replay_batch_size : replay에서 샘플링할 rollout 개수

truncation_clip : importance ratio clipping 값

# 8. ACER 핵심 수식 직관
### 8.1 Importance Ratio

과거 정책과 현재 정책의 차이를 보정하기 위해 사용합니다.

rho = pi(a|s) / mu(a|s)

pi(a|s) : 현재 정책

mu(a|s) : 데이터를 수집할 당시의 정책

### 8.2 Truncated Importance Sampling

importance ratio가 너무 크면 분산이 커질 수 있으므로 clip을 적용합니다.

rho_bar = min(rho, c)
### 8.3 Bias Correction

clip 과정에서 생길 수 있는 편향을 줄이기 위해 추가 보정항을 둡니다.

### 8.4 Retrace

off-policy 상황에서도 비교적 안정적인 return target을 계산하기 위한 방식입니다.

# 9. 한계점

이 구현은 교육용 버전이므로 다음과 같은 한계가 있습니다.

논문의 모든 기법을 완전히 구현한 것은 아님

trust region 업데이트는 생략됨

stochastic dueling architecture는 생략됨

CartPole에 맞춘 단순한 MLP 구조만 사용

Atari와 같은 고차원 입력 환경용 CNN 구조는 포함되지 않음

즉, 이 코드는 ACER의 핵심 아이디어를 이해하고 직접 실행해보기 위한 입문용 구현입니다.

---

# 10. 결론

ACER의 핵심 아이디어인 Replay Buffer 기반 데이터 재사용과 Off-policy 보정을 직접 구현해본 예제입니다.

이를 통해 다음 내용을 학습할 수 있습니다.

Actor-Critic 구조

on-policy와 off-policy의 차이

Experience Replay의 장단점

Importance Sampling의 필요성

Retrace와 Bias Correction의 역할

강화학습에서 ACER는 구현 난이도가 다소 높지만,
왜 actor-critic에 replay를 붙이는 것이 어려운지를 이해하는 데 매우 좋은 알고리즘입니다.
