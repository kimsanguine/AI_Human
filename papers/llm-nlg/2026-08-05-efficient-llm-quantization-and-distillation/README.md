# Daily AI Paper Recommendations

> **Date:** 2026-08-05
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Efficient LLM Quantization and Distillation

---

## Paper 1 (Classic): Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding
- **Authors:** Song Han, Huizi Mao, William J. Dally
- **Year:** 2015 (ICLR 2016 Best Paper)
- **arXiv:** https://arxiv.org/abs/1510.00149
- **PDF:** [./deep-compression-han-2015.pdf](./deep-compression-han-2015.pdf)
- **Citation Count:** ~10,000+ (approximate)

### 요약
Deep Compression은 프루닝(pruning), 학습된 양자화(trained quantization), 허프만 코딩(Huffman coding)의 3단계 파이프라인으로 신경망의 저장 용량을 정확도 손실 없이 35~49배 줄이는 방법을 제안한다. AlexNet을 240MB에서 6.9MB로(35배), VGG-16을 552MB에서 11.3MB로(49배) 압축하면서도 정확도를 유지했다. 모델 압축 분야 전체의 기틀을 세운 논문이다.

### 핵심 기여
- 중요한 연결만 남기는 프루닝 → 가중치 공유 기반 양자화 → 허프만 코딩으로 이어지는 3단계 통합 압축 파이프라인 제시
- 가중치 클러스터링과 재학습(retraining)을 통해 극단적인 압축에서도 정확도를 보존하는 기법 확립
- 온칩 SRAM에 모델을 올릴 수 있을 만큼 크기를 줄여 에너지 효율과 추론 속도를 크게 개선

### 이 논문이 중요한 이유
오늘날 LLM 양자화(GPTQ, AWQ, QLoRA 등)의 사고 방식 대부분이 이 논문의 "가중치는 중복이 많고, 정밀도를 낮춰도 재학습으로 회복 가능하다"는 통찰에서 출발한다. AI 엔지니어가 압축·양자화의 근본 원리를 이해하려면 반드시 거쳐야 할 출발점이다.

### 사전 지식
신경망 가중치 표현과 float32/int8 정밀도 개념, 프루닝의 기본 아이디어, k-means 클러스터링, 엔트로피 코딩(허프만) 정도를 알면 읽기 수월하다.

### 관련 논문
- [Learning both Weights and Connections for Efficient Neural Networks (Han et al., 2015)](https://arxiv.org/abs/1506.02626)
- [The Lottery Ticket Hypothesis (Frankle & Carbin, 2018)](https://arxiv.org/abs/1803.03635)

### 실무 적용
모바일/엣지 디바이스 온디바이스 추론, TensorRT·CoreML 등에서의 모델 경량화, LLM 서빙 비용 절감을 위한 가중치 압축 전략의 개념적 토대로 활용된다. "먼저 프루닝, 그다음 양자화, 마지막에 인코딩"이라는 실무 압축 워크플로우의 원형이다.

---

## Paper 2 (Classic): Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
- **Authors:** Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko
- **Year:** 2017 (CVPR 2018)
- **arXiv:** https://arxiv.org/abs/1712.05877
- **PDF:** [./integer-arithmetic-only-inference-jacob-2017.pdf](./integer-arithmetic-only-inference-jacob-2017.pdf)
- **Citation Count:** ~6,000+ (approximate)

### 요약
이 논문은 추론을 부동소수점 없이 정수(int8) 연산만으로 수행할 수 있게 하는 양자화 스킴과, 이를 위한 양자화 인식 학습(Quantization-Aware Training, QAT) 절차를 함께 제안한다. 정수 전용 하드웨어에서 정확도 손실을 최소화하면서 지연시간과 모델 크기를 동시에 개선하며, 오늘날 int8 배포의 사실상 표준이 된 대칭/비대칭 어파인 양자화 공식을 정립했다.

### 핵심 기여
- 실수값을 정수와 스케일·제로포인트로 표현하는 어파인 양자화(affine quantization) 수식을 명확히 정의
- 학습 시 양자화 효과를 시뮬레이션하는 "가짜 양자화(fake quantization)" 노드를 통한 QAT 기법 제안
- 정수 전용 행렬 곱셈으로 추론을 수행하는 방법을 제시하여 정확도-지연시간 트레이드오프를 실측으로 검증

### 이 논문이 중요한 이유
LLM을 포함한 현대 모델의 int8/int4 배포에서 쓰이는 스케일·제로포인트 기반 양자화의 이론적 근간이다. Post-Training Quantization(PTQ)과 QAT의 차이를 이해하고, 왜 단순 반올림이 아니라 학습이 필요한지를 체득하는 데 필수적이다.

### 사전 지식
고정소수점(fixed-point)과 부동소수점의 차이, 행렬 곱셈의 연산 비용, 순전파/역전파 기본, straight-through estimator 개념을 알면 도움이 된다.

### 관련 논문
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs (Xiao et al., 2022)](https://arxiv.org/abs/2211.10438)

### 실무 적용
TensorFlow Lite, PyTorch의 int8 양자화, ONNX Runtime, 엣지 TPU/모바일 NPU 배포에서 그대로 쓰이는 표준 양자화 방식이다. LLM 서빙에서 int8 KV-cache나 weight-only 양자화를 설계할 때의 기본 문법을 제공한다.

---

## Paper 3 (Recent): Extreme Compression of Large Language Models via Additive Quantization (AQLM)
- **Authors:** Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, Dan Alistarh
- **Year:** 2024 (ICML 2024)
- **arXiv:** https://arxiv.org/abs/2401.06118
- **PDF:** [./aqlm-additive-quantization-egiazarian-2024.pdf](./aqlm-additive-quantization-egiazarian-2024.pdf)
- **Citation Count:** ~200+ (approximate)

### 요약
AQLM(Additive Quantization of Language Models)은 정보 검색 분야의 고전인 Additive Quantization(AQ)을 LLM 가중치 압축에 적용하여, 파라미터당 2~3비트라는 극단적 압축 영역에서 최초로 정확도-크기 파레토 최적을 달성한 방법이다. 여러 코드북의 합으로 가중치를 표현하고, 트랜스포머 블록 단위로 코드북을 공동 최적화하여 2비트에서 기존 모든 기법을 능가한다.

### 핵심 기여
- 학습된 가법 양자화(learned additive quantization)를 입력 적응적으로 적용해 가중치 행렬을 압축
- 블록 단위 코드북 공동 최적화로 2비트 극저비트 영역에서 정확도 열화를 크게 완화
- FP16에 근접하거나 능가하는 GPU/CPU 토큰 생성 커널을 제공하여 실사용 속도까지 확보

### 이 논문이 중요한 이유
GPTQ·AWQ가 3~4비트에서 강했다면, AQLM은 2비트라는 한계 영역을 실용 가능하게 끌어올려 대형 모델을 소비자용 GPU/노트북에서 돌릴 수 있게 만든다. "얼마나 더 압축할 수 있는가"의 최전선을 보여주는 2024년 대표작이다.

### 사전 지식
Paper 1·2의 양자화 기본기, 벡터 양자화(vector quantization)와 코드북/코드워드 개념, GPTQ의 오차 보정 아이디어를 알면 이해가 빠르다.

### 관련 논문
- [QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks (Tseng et al., 2024)](https://arxiv.org/abs/2402.04396)
- [PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression (Malinovskii et al., 2024)](https://arxiv.org/abs/2405.14852)

### 실무 적용
메모리 제약이 큰 환경에서 70B급 모델을 단일 GPU에 올리는 극한 양자화 배포, 온디바이스 LLM, 서빙 비용 최소화 시나리오에 적용된다. vLLM·transformers 등에 통합되어 2비트 가중치 모델을 실제로 서빙할 수 있다.

---

## 추천 읽기 순서
1. **Deep Compression (2015)** — 압축의 큰 그림(프루닝+양자화+인코딩)을 먼저 잡는다.
2. **Integer-Arithmetic-Only Inference (2017)** — 양자화의 수학적 정의(스케일·제로포인트)와 QAT를 익힌다.
3. **AQLM (2024)** — 앞의 원리가 LLM의 극한 압축(2비트)까지 어떻게 진화했는지 확인한다.

## 핵심 테이크어웨이
- 신경망 가중치는 중복이 많아, 정밀도를 크게 낮춰도 재학습/보정으로 정확도를 회복할 수 있다.
- 양자화의 핵심 문법은 스케일과 제로포인트로 실수를 정수로 매핑하는 것이며, QAT와 PTQ는 이 오차를 다루는 방식이 다르다.
- LLM 시대의 최전선은 weight-only 초저비트(2~3비트) 양자화이며, 코드북 기반 방법(AQLM)이 파레토 프론티어를 밀어내고 있다.

## 다음 토픽과의 연결
다음 모듈(Module 7: Prompt Engineering)에서는 모델 자체를 바꾸지 않고 입력(프롬프트)으로 성능을 끌어내는 방향으로 넘어간다. 오늘의 양자화·압축이 "모델을 싸게 돌리는 법"이라면, 다음은 "싸게 돌리는 모델에서 더 좋은 답을 끌어내는 법"으로 이어진다.
