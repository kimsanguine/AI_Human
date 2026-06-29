# Syllabus Essential (교안 보강): Generative Adversarial Nets

> **Category:** ml-deep-learning / generative-models
> **Why added:** 교안(CNN Architectures 노트북, SD_FLUX)에서 비중 있게 다루지만 repo에 생성모델 classic이 없어 보강
> **Type:** Classic (Must-Read)

## Paper: Generative Adversarial Nets
- **Authors:** Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1406.2661
- **PDF:** [./generative-adversarial-nets-goodfellow-2014.pdf](./generative-adversarial-nets-goodfellow-2014.pdf)
- **Citation Count:** 약 70,000+ (Google Scholar 기준)

### 요약
생성자(G)와 판별자(D) 두 네트워크를 적대적으로 동시에 학습시키는 생성 모델 프레임워크를 제안한다. G는 데이터 분포를 모사하고 D는 입력이 진짜인지 G가 만든 가짜인지 판별하며, 둘은 minimax 게임을 벌인다. 마르코프 체인이나 근사 추론 없이 역전파만으로 학습 가능하다는 점이 혁신이다.

### 핵심 기여
- 적대적 학습(adversarial training)이라는 새로운 생성 모델 패러다임 정립
- 이론적으로 최적해에서 G가 실제 데이터 분포를 복원하고 D=1/2가 됨을 증명
- 별도의 추론 네트워크·마르코프 체인 없이 backprop만으로 학습하는 간결한 구조

### 이 논문이 중요한 이유
DCGAN, StyleGAN, Pix2Pix, CycleGAN, 그리고 오늘날 확산모델(Diffusion) 이전 생성 AI의 10년을 연 출발점이다. "두 네트워크의 경쟁으로 분포를 학습한다"는 사고는 이후 적대적 손실(adversarial loss)로 TTS(HiFi-GAN), 초해상도, 이미지 편집 전반에 스며들었다. 생성 AI를 가르치는 교안의 토대 논문이다.

### 사전 지식
- MLP와 역전파, 확률분포·KL/JS divergence 기초
- minimax 최적화와 내쉬 균형 개념
- 우도(likelihood) 기반 생성 모델과의 차이

### 관련 논문
- [Unsupervised Representation Learning with Deep Convolutional GANs / DCGAN (Radford et al., 2015)](https://arxiv.org/abs/1511.06434)
- [Auto-Encoding Variational Bayes / VAE (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)
- [Image-to-Image Translation with Conditional Adversarial Networks / Pix2Pix (Isola et al., 2016)](https://arxiv.org/abs/1611.07004)

### 실무 적용
데이터 증강, 이미지 생성·편집·복원, 음성 합성의 보코더(HiFi-GAN), 도메인 변환 등에 폭넓게 쓰인다. 확산모델이 주류가 된 지금도 적대적 손실은 품질·속도 튜닝 도구로 결합되며, GAN의 학습 불안정성(mode collapse) 문제를 이해하는 것은 모든 생성 AI 디버깅의 기본기다.
