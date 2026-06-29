# Syllabus Essential (교안 보강): MobileNets

> **Category:** ml-deep-learning / cnn-architectures
> **Why added:** 교안(CNN)에서 경량 CNN 사례로 인용되지만 repo에 전용 항목이 없어 보강
> **Type:** Classic (Must-Read)

## Paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
- **Authors:** Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
- **Year:** 2017
- **arXiv:** https://arxiv.org/abs/1704.04861
- **PDF:** [./mobilenets-howard-2017.pdf](./mobilenets-howard-2017.pdf)
- **Citation Count:** 약 25,000+ (Google Scholar 기준)

### 요약
depthwise separable convolution(채널별 공간 합성곱 + 1×1 점별 합성곱)을 핵심으로 한 경량 CNN을 제안한다. width multiplier와 resolution multiplier라는 두 전역 하이퍼파라미터로 지연-정확도를 손쉽게 조절해, 모바일·임베디드 환경에 맞는 모델 크기를 선택할 수 있게 한다.

### 핵심 기여
- depthwise separable convolution으로 연산량·파라미터를 대폭 절감
- width/resolution multiplier로 모델 크기를 연속적으로 조절하는 설계 제시
- 정확도-자원 트레이드오프를 ImageNet 등에서 광범위하게 실증

### 이 논문이 중요한 이유
"성능을 거의 유지하며 모델을 가볍게"라는 엣지 AI의 실무 과제를 정면으로 다룬 대표작이다. depthwise separable conv는 이후 EfficientNet(MBConv), Xception 등 효율 아키텍처의 기본 부품이 됐다. 온디바이스 추론을 가르치는 교안의 핵심 사례다.

### 사전 지식
- 표준 합성곱의 연산량 계산(커널×채널)
- 1×1 convolution의 역할
- 정확도-지연-메모리 트레이드오프 개념

### 관련 논문
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al., 2018)](https://arxiv.org/abs/1801.04381)
- [Xception: Deep Learning with Depthwise Separable Convolutions (Chollet, 2016)](https://arxiv.org/abs/1610.02357)
- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)

### 실무 적용
모바일 앱, IoT, 실시간 카메라 추론 등 자원 제약 환경의 비전 모델 기본 백본이다. width/resolution multiplier로 타깃 하드웨어 예산에 맞춰 모델을 재단하는 방식은 오늘날 엣지 배포 파이프라인의 표준 패턴이다.
