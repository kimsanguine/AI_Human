# Syllabus Essential (교안 보강): Auto-Encoding Variational Bayes

> **Category:** ml-deep-learning / generative-models
> **Why added:** 생성모델 양대 classic 중 하나. GAN과 짝을 이뤄 교안 생성모델 파트를 보강
> **Type:** Classic (Must-Read)

## Paper: Auto-Encoding Variational Bayes (VAE)
- **Authors:** Diederik P. Kingma, Max Welling
- **Year:** 2013
- **arXiv:** https://arxiv.org/abs/1312.6114
- **PDF:** [./auto-encoding-variational-bayes-kingma-2013.pdf](./auto-encoding-variational-bayes-kingma-2013.pdf)
- **Citation Count:** 약 40,000+ (Google Scholar 기준)

### 요약
연속 잠재변수를 가진 방향성 확률모델에서 사후분포(posterior)가 다루기 어려울 때, 변분 하한(ELBO)을 재매개변수화(reparameterization trick)해 표준 SGD로 최적화하는 방법(AEVB)을 제안한다. 신경망을 인식 모델로 쓰면 우리가 아는 VAE가 된다.

### 핵심 기여
- 재매개변수화 트릭으로 잠재변수 샘플링을 미분 가능하게 만들어 backprop 학습 가능
- 인코더(인식 모델)+디코더(생성 모델) 구조로 효율적 근사 추론 구현
- 변분추론과 딥러닝을 결합한 확률적 생성모델의 표준 틀 제시

### 이 논문이 중요한 이유
GAN과 더불어 딥 생성모델의 양대 축이다. 잠재공간(latent space)을 명시적으로 모델링한다는 점에서 표현 학습·이상탐지·반지도학습으로 확장성이 크고, 무엇보다 Stable Diffusion의 핵심 부품인 VAE(잠재 압축)와 확산모델의 변분 관점으로 직결된다. 교안의 SD_FLUX 파트를 이해하려면 필수 선행 지식이다.

### 사전 지식
- 오토인코더 구조, 베이즈 정리와 사후분포
- KL divergence, 변분추론(variational inference)과 ELBO
- 정규분포 재매개변수화(평균+표준편차×noise)

### 관련 논문
- [Generative Adversarial Nets (Goodfellow et al., 2014)](https://arxiv.org/abs/1406.2661)
- [High-Resolution Image Synthesis with Latent Diffusion Models / Stable Diffusion (Rombach et al., 2021)](https://arxiv.org/abs/2112.10752)
- [beta-VAE: Learning Basic Visual Concepts (Higgins et al., 2017)](https://openreview.net/forum?id=Sy2fzU9gl)

### 실무 적용
잠재공간 압축(예: Stable Diffusion의 latent VAE), 이상탐지, 데이터 생성·보간, 표현 학습에 쓰인다. 확산모델 파이프라인에서 이미지를 latent로 인코딩/디코딩하는 단계가 바로 VAE이며, 잠재공간의 연속성과 KL 정규화의 트레이드오프를 이해하는 것이 생성 품질 튜닝의 핵심이다.
