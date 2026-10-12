# Daily AI Paper Recommendations

> **Date:** 2026-10-13
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Optimization and Regularization

---

## Paper 1 (Classic): On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima
- **Authors:** Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, Ping Tak Peter Tang
- **Year:** 2016 (ICLR 2017)
- **arXiv:** https://arxiv.org/abs/1609.04836
- **PDF:** [./sharp-minima-large-batch-keskar-2016.pdf](./sharp-minima-large-batch-keskar-2016.pdf)
- **Citation Count:** ~4,000+

### 요약
배치 크기를 키우면 학습은 빨라지지만 테스트 성능이 떨어지는 "일반화 격차(generalization gap)"가 왜 생기는지를 실험적으로 파고든 논문이다. 큰 배치(Large-Batch)로 학습하면 손실 곡면의 뾰족한(sharp) 최소점으로 수렴하고, 작은 배치(Small-Batch)는 그래디언트 노이즈 덕분에 평평한(flat) 최소점으로 수렴한다는 것을 보였다. 뾰족한 최소점은 학습/테스트 분포의 작은 차이에도 손실이 크게 변해 일반화가 나빠진다.

### 핵심 기여
- 대규모 배치 학습의 일반화 격차(작은 네트워크에서도 최대 ~5%)를 체계적으로 실증
- "sharp vs. flat minima"라는 프레임으로 일반화 현상을 설명하고, 이를 정량화하는 sharpness 지표 제안
- 1D 파라미터 보간(interpolation) 시각화로 두 해의 곡면 모양 차이를 직관적으로 보여줌
- Warm-start, 데이터 증강 등 격차를 줄이려는 초기 처방과 열린 질문을 정리

### 이 논문이 중요한 이유
"GPU를 더 붙여서 배치를 키우면 그냥 빨라지는 것 아닌가?"라는 질문에 대한 가장 유명한 반례다. 이후 등장한 Linear Scaling Rule + Warmup(Goyal 2017), LAMB(You 2019), SAM(Foret 2020) 등은 모두 이 논문이 던진 문제를 해결하려는 시도로 읽을 수 있다. 분산 학습 설정을 결정하는 AI 엔지니어라면 배치 크기가 단순한 처리량 파라미터가 아니라 **일반화에 영향을 주는 정규화 변수**라는 관점을 반드시 가져야 한다.

### 사전 지식
- 미니배치 SGD와 그래디언트 노이즈(분산)의 개념
- 헤시안(Hessian)과 곡률(curvature)의 직관적 의미
- 학습/검증 손실 곡선으로 과적합·일반화를 진단하는 방법

### 관련 논문
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (Goyal et al., 2017)](https://arxiv.org/abs/1706.02677)
- [Sharpness-Aware Minimization for Efficiently Improving Generalization (Foret et al., 2020)](https://arxiv.org/abs/2010.01412)
- [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes / LAMB (You et al., 2019)](https://arxiv.org/abs/1904.00962)
- [An Empirical Model of Large-Batch Training (McCandlish et al., 2018)](https://arxiv.org/abs/1812.06162)

### 실무 적용
멀티 GPU로 학습을 옮길 때 "배치를 8배 키웠더니 정확도가 떨어졌다"는 문제는 지금도 매우 흔하다. 실무에서는 (1) 학습률을 배치에 비례해 키우고 warmup을 넣거나, (2) gradient accumulation으로 유효 배치를 조절하거나, (3) critical batch size(McCandlish 2018) 이상으로는 배치를 늘리지 않는 방식으로 대응한다. LLM 파인튜닝에서도 소량 데이터에 큰 배치를 쓰면 쉽게 sharp한 해로 빠지므로, 작은 배치 + 낮은 학습률이 오히려 안정적인 경우가 많다.

---

## Paper 2 (Classic): When Does Label Smoothing Help?
- **Authors:** Rafael Müller, Simon Kornblith, Geoffrey Hinton
- **Year:** 2019 (NeurIPS 2019)
- **arXiv:** https://arxiv.org/abs/1906.02629
- **PDF:** [./label-smoothing-muller-2019.pdf](./label-smoothing-muller-2019.pdf)
- **Citation Count:** ~2,500+

### 요약
정답 레이블을 one-hot(1.0) 대신 0.9 같은 "부드러운" 값으로 바꾸는 Label Smoothing이 왜, 언제 효과가 있는지를 분석한 논문이다. Label Smoothing은 같은 클래스의 표현(penultimate layer)을 촘촘한 군집으로 모으고, 모델의 확률 보정(calibration)을 개선해 빔 서치 품질까지 높인다. 반면 클래스 간 유사도 정보가 logit에서 지워지기 때문에, 이렇게 학습된 teacher로는 지식 증류(Knowledge Distillation)가 훨씬 덜 효과적이라는 반전 결과를 제시했다.

### 핵심 기여
- Penultimate layer 표현을 2D로 투영하는 시각화 기법으로 Label Smoothing의 효과를 직관적으로 설명
- Label Smoothing이 일반화뿐 아니라 **calibration(ECE 감소)**을 개선한다는 점을 실증, Temperature Scaling과 유사한 효과 확인
- Label Smoothing으로 학습한 teacher는 증류 성능을 떨어뜨린다는 실무적으로 중요한 트레이드오프 발견
- 정규화 기법을 "정확도 +α"가 아닌 "표현 공간을 어떻게 바꾸는가"로 분석하는 관점 제시

### 이 논문이 중요한 이유
Label Smoothing은 Inception-v2, Transformer(Attention Is All You Need), 대부분의 이미지 분류 레시피에 기본으로 들어가 있어 "그냥 켜두는 옵션"이 되기 쉽다. 이 논문은 정규화 기법이 공짜가 아니며, **어떤 정보를 버리는 대가로 무엇을 얻는지** 이해해야 한다는 것을 보여준다. 특히 대형 모델을 소형 모델로 증류해 서비스하는 AI 제품 팀이라면 teacher 학습 레시피에서 Label Smoothing을 끌지 말지 반드시 알아야 한다.

### 사전 지식
- Cross-entropy 손실과 softmax, one-hot 레이블
- 모델 calibration(ECE, reliability diagram)과 temperature scaling
- Knowledge Distillation(Hinton et al., 2015)의 soft target 개념

### 관련 논문
- [Rethinking the Inception Architecture for Computer Vision (Szegedy et al., 2015)](https://arxiv.org/abs/1512.00567)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [On Calibration of Modern Neural Networks (Guo et al., 2017)](https://arxiv.org/abs/1706.04599)

### 실무 적용
PyTorch에서는 `nn.CrossEntropyLoss(label_smoothing=0.1)` 한 줄로 적용할 수 있다. 분류 모델의 confidence 점수를 그대로 비즈니스 로직(예: 임계값 기반 자동 승인, 휴먼 리뷰 라우팅)에 쓰는 경우 calibration 개선 효과가 크다. 반대로 teacher→student 증류 파이프라인(예: 대형 ASR/번역 모델을 온디바이스 모델로 압축)에서는 teacher 학습 시 Label Smoothing을 끄는 것이 유리할 수 있다. 번역·음성 인식처럼 빔 서치를 쓰는 생성 모델에서도 기본 정규화 기법으로 여전히 많이 쓰인다.

---

## Paper 3 (Recent): SOAP: Improving and Stabilizing Shampoo using Adam
- **Authors:** Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
- **Year:** 2024 (ICLR 2025)
- **arXiv:** https://arxiv.org/abs/2409.11321
- **PDF:** [./soap-shampoo-adam-vyas-2024.pdf](./soap-shampoo-adam-vyas-2024.pdf)
- **Citation Count:** ~200+

### 요약
2차(second-order) 정보를 활용하는 Shampoo 옵티마이저가 사실상 "프리컨디셔너의 고유기저(eigenbasis)에서 Adafactor를 돌리는 것"과 같다는 이론적 연결을 밝히고, 이를 바탕으로 **Shampoo의 고유기저 안에서 Adam을 실행하는** SOAP을 제안했다. 비싼 고유값 분해는 일정 주기마다만 수행하고, 그 사이에는 회전된 좌표계에서 Adam의 1·2차 모멘트를 계속 업데이트한다. 360M/660M 언어 모델 사전학습에서 AdamW 대비 반복 횟수 약 40%, 실제 학습 시간(wall-clock) 약 35%를 줄였다고 보고한다.

### 핵심 기여
- Shampoo ↔ Adafactor(고유기저 회전) 사이의 수학적 등가성 정립
- Adam 대비 추가 하이퍼파라미터가 사실상 "프리컨디셔닝 주기" 하나뿐인 단순한 설계
- 프리컨디셔닝 주기를 늘려도 성능 저하가 Shampoo보다 훨씬 작아 연산 효율 확보
- 대규모 LM 사전학습에서 AdamW·Shampoo 대비 반복 수와 wall-clock 모두 개선을 실증

### 이 논문이 중요한 이유
2024–2025년 옵티마이저 연구의 핵심 흐름은 "Adam의 대각(diagonal) 근사를 넘어, 행렬 구조를 쓰는 프리컨디셔닝"이다(Shampoo가 AlgoPerf 벤치마크에서 우승, Muon이 대규모 LLM 학습에 채택). SOAP은 그 흐름의 대표 논문으로, Adam과 2차 방법 사이의 다리를 명확히 보여준다. 오늘의 클래식 두 편이 "어디로 수렴하는가(일반화)"를 다룬다면, SOAP은 "얼마나 빨리, 싸게 도달하는가(효율)"를 다뤄 최적화의 두 축을 완성한다.

### 사전 지식
- Adam/AdamW의 1차·2차 모멘트 업데이트 방식
- 프리컨디셔닝, 고유값 분해(eigendecomposition)의 기본 개념
- Shampoo, Adafactor 옵티마이저의 대략적 아이디어
- (선택) 같은 저장소의 Muon 논문(2026-09-16)과 비교하며 읽으면 좋음

### 관련 논문
- [Shampoo: Preconditioned Stochastic Tensor Optimization (Gupta et al., 2018)](https://arxiv.org/abs/1802.09568)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (Shazeer & Stern, 2018)](https://arxiv.org/abs/1804.04235)
- [Muon is Scalable for LLM Training (Liu et al., 2025)](https://arxiv.org/abs/2502.16982)
- [Understanding SOAP from the Perspective of Gradient Whitening (2025)](https://arxiv.org/abs/2509.22938)

### 실무 적용
자체 도메인 모델(예: 음성/더빙용 수억 파라미터급 모델)을 사전학습하거나 대규모 지속학습을 할 때, 옵티마이저 교체만으로 GPU 시간을 30% 이상 줄일 수 있다는 점은 비용 측면에서 매우 크다. 공개 구현(github.com/nikhilvyas/SOAP)을 AdamW 대신 바로 넣어 실험할 수 있으며, 메모리 오버헤드(행렬 프리컨디셔너 저장)가 있으므로 매우 큰 임베딩/출력층에는 AdamW를 유지하는 하이브리드 구성이 흔하다. 단, 소규모 파인튜닝(LoRA 등)에서는 이득이 작으므로 사전학습·대규모 학습에 우선 적용하는 것이 합리적이다.

---

## 추천 읽기 순서
1. **Keskar et al. (2016)** — 배치 크기와 sharp/flat minima로 "최적화가 일반화를 결정한다"는 큰 그림부터 잡는다.
2. **Müller et al. (2019)** — 손실 함수 수준의 정규화(Label Smoothing)가 표현 공간과 calibration을 어떻게 바꾸는지 본다.
3. **Vyas et al. (2024, SOAP)** — 일반화에서 효율로 관점을 옮겨, Adam 이후 차세대 옵티마이저의 설계 원리를 이해한다.

## 핵심 테이크어웨이
- **배치 크기는 정규화 변수다.** 큰 배치는 그래디언트 노이즈를 줄여 sharp minima로 이끌 수 있으므로, 학습률·warmup과 함께 설계해야 한다.
- **정규화에는 대가가 있다.** Label Smoothing은 일반화·calibration을 얻는 대신 클래스 간 유사도 정보를 버리며, 이는 증류 성능 저하로 이어진다.
- **옵티마이저의 다음 단계는 "구조를 쓰는 프리컨디셔닝"이다.** SOAP·Shampoo·Muon은 Adam의 대각 근사를 넘어 행렬 구조를 활용해 학습 비용을 크게 줄인다.
- 질문으로 정리하면: "우리 모델은 *어디로* 수렴하고(일반화), *얼마나 싸게* 도달하며(효율), 그 과정에서 *무엇을 잃는가*(트레이드오프)?"

## 다음 토픽과의 연결
다음 토픽은 **Transfer Learning and Foundation Models**다. 오늘 다룬 최적화·정규화 기법은 대규모 사전학습(SOAP, 대배치 학습)과 다운스트림 파인튜닝(작은 배치, Label Smoothing, 증류) 양쪽에서 그대로 쓰인다. 특히 "사전학습된 모델이 flat한 영역에 있을수록 전이(transfer)가 잘 된다"는 관점은 파운데이션 모델의 전이 성능을 이해하는 데 오늘의 Keskar 논문과 직접 연결된다.
