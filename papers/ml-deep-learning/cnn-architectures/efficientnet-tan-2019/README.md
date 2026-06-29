# Syllabus Essential (교안 보강): EfficientNet

> **Category:** ml-deep-learning / cnn-architectures
> **Why added:** 교안(CNN, ViT_Clip)에서 반복 인용되지만 repo에 전용 항목이 없어 CNN 스케일링 공백을 보강
> **Type:** Classic (Must-Read)

## Paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
- **Authors:** Mingxing Tan, Quoc V. Le
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1905.11946
- **PDF:** [./efficientnet-tan-2019.pdf](./efficientnet-tan-2019.pdf)
- **Citation Count:** 약 15,000+ (Google Scholar 기준)

### 요약
CNN의 깊이(depth)·너비(width)·입력 해상도(resolution)를 제각각 키우던 관행을 대신해, 하나의 복합 계수(compound coefficient)로 세 차원을 균형 있게 동시에 스케일링하는 방법을 제안한다. NAS로 찾은 베이스라인(EfficientNet-B0)을 키워 B7에서 ImageNet top-1 84.3%를 달성하면서도 기존 최고 모델보다 8.4배 작고 6.1배 빠르다.

### 핵심 기여
- depth/width/resolution을 함께 키우는 compound scaling 법칙 정립
- NAS 기반 효율적 베이스라인 EfficientNet-B0 설계
- 정확도-효율 파레토 프론티어를 크게 끌어올린 모델 패밀리(B0~B7) 제시

### 이 논문이 중요한 이유
"무작정 크게"가 아니라 "균형 있게 스케일링"이라는 사고를 정립해, 이후 모델 크기 설계의 표준 사고법이 됐다. 제한된 연산/메모리에서 최적 모델을 고르는 실무 의사결정과 직결되며, EfficientNetV2·ConvNeXt 등 후속 효율 아키텍처의 출발점이다.

### 사전 지식
- CNN 기본 구조와 ResNet/MobileNet의 블록(특히 depthwise separable conv, MBConv)
- Neural Architecture Search(NAS) 개념
- FLOPs·파라미터 수와 정확도의 트레이드오프

### 관련 논문
- [MobileNets (Howard et al., 2017)](https://arxiv.org/abs/1704.04861)
- [Deep Residual Learning for Image Recognition / ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [EfficientNetV2: Smaller Models and Faster Training (Tan & Le, 2021)](https://arxiv.org/abs/2104.00298)

### 실무 적용
온디바이스/엣지 추론, 비용 제약 하의 비전 서비스에서 정확도-지연 균형점을 고를 때 1순위 후보다. 전이학습 백본으로도 널리 쓰이며, "모델을 얼마나, 어느 차원으로 키울지"를 데이터·예산 기준으로 결정하는 사고 프레임을 제공한다.
