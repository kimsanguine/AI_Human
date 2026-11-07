# Daily AI Paper Recommendations

> **Date:** 2026-11-08
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** RNN LSTM and Sequence Models

> 이번 회차는 이전 회차(LSTM, Seq2Seq, GRU, NTM, Graves, Pascanu, S4, Mamba, HiPPO, LSTM Odyssey, Zaremba 등)와 겹치지 않도록 **"RNN 내부를 들여다보기 → Attention과 RNN의 통합 → 은닉 상태를 학습기로 확장"** 이라는 흐름으로 새 논문을 골랐습니다.

---

## Paper 1 (Classic): Visualizing and Understanding Recurrent Networks
- **Authors:** Andrej Karpathy, Justin Johnson, Li Fei-Fei
- **Year:** 2015 (ICLR 2016 Workshop)
- **arXiv:** https://arxiv.org/abs/1506.02078
- **PDF:** [./visualizing-understanding-rnns-karpathy-2015.pdf](./visualizing-understanding-rnns-karpathy-2015.pdf)
- **Citation Count:** ~1,500+

### 요약
문자 단위 언어 모델(char-level LM)로 학습한 LSTM/GRU의 내부 셀을 시각화해, 따옴표·괄호·줄 길이 같은 장거리 구조를 추적하는 "해석 가능한 셀"이 실제로 존재함을 보여준 논문입니다. 또한 n-gram 모델과의 오류 비교를 통해 LSTM이 어떤 종류의 장거리 의존성에서 이득을 보는지 정량적으로 분해합니다.

### 핵심 기여
- 따옴표 내부 여부, 줄 위치, 코드 들여쓰기 깊이 등을 추적하는 개별 셀을 발견하고 시각화
- 게이트 포화(saturation) 통계 분석으로 LSTM 게이트가 실제로 "켜짐/꺼짐" 스위치처럼 동작함을 확인
- n-gram/n-NN 기준선 대비 오류 유형 분해(error analysis) — 남은 오류가 어디서 오는지 체계화

### 이 논문이 중요한 이유
"모델이 무엇을 기억하는가?"라는 질문을 데이터로 답한 초기 해석 가능성(interpretability) 연구의 대표작입니다. 오늘날 LLM의 뉴런/피처 분석(mechanistic interpretability)의 사고방식이 여기서 시작되었고, 모델을 블랙박스가 아닌 "디버깅 가능한 시스템"으로 보는 엔지니어 관점을 길러줍니다.

### 사전 지식
RNN/LSTM/GRU의 게이트 구조, 문자 단위 언어 모델, cross-entropy와 perplexity 개념

### 관련 논문
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (Chung et al., 2014)](https://arxiv.org/abs/1412.3555)
- [Toy Models of Superposition (Elhage et al., 2022)](https://arxiv.org/abs/2209.10652)

### 실무 적용
시퀀스 모델이 이상한 출력을 낼 때, 은닉 상태 활성화를 시각화해 "어떤 정보가 언제 사라지는지" 추적하는 디버깅 기법으로 그대로 쓰입니다. 음성·로그 이상 탐지 등에서 모델 설명 자료(고객/규제 대응)를 만들 때도 유용합니다.

---

## Paper 2 (Classic): Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- **Authors:** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret
- **Year:** 2020 (ICML 2020)
- **arXiv:** https://arxiv.org/abs/2006.16236
- **PDF:** [./transformers-are-rnns-linear-attention-katharopoulos-2020.pdf](./transformers-are-rnns-linear-attention-katharopoulos-2020.pdf)
- **Citation Count:** ~2,500+

### 요약
softmax 어텐션을 커널 특징 맵(feature map)의 내적으로 바꾸면, 행렬 곱 순서를 바꿔 계산 복잡도를 O(N²)에서 O(N)으로 줄일 수 있음을 보였습니다. 더 나아가 causal linear attention은 "행렬 형태의 은닉 상태를 누적 갱신하는 RNN"과 수학적으로 동일하다는 것을 증명해, 자기회귀 추론을 최대 수천 배 가속했습니다.

### 핵심 기여
- 커널 기반 linear attention (feature map φ(x)=elu(x)+1) 제안
- Causal Transformer = 행렬 은닉 상태(S = Σ φ(k)vᵀ)를 가진 RNN이라는 등가성 증명
- 긴 시퀀스 이미지 생성·음성 인식에서 최대 4000배 빠른 자기회귀 추론

### 이 논문이 중요한 이유
Transformer와 RNN이 "다른 세계"가 아니라 같은 스펙트럼 위에 있다는 관점을 정립한 논문입니다. 이후 RetNet, GLA, Mamba-2(SSD), DeltaNet, RWKV, TTT 등 거의 모든 "선형 시간 시퀀스 모델"이 이 프레임 위에서 설명됩니다. 현대 아키텍처 논문을 읽기 위한 공통 언어입니다.

### 사전 지식
Scaled dot-product attention, 커널 트릭 개념, 자기회귀 디코딩과 KV cache

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Rethinking Attention with Performers (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794)
- [Gated Linear Attention Transformers with Hardware-Efficient Training (Yang et al., 2023)](https://arxiv.org/abs/2312.06635)

### 실무 적용
긴 컨텍스트 스트리밍(실시간 음성 인식, 온디바이스 추론)처럼 KV cache 메모리가 병목인 환경에서 고정 크기 상태로 추론하는 설계의 근거가 됩니다. 하이브리드 모델(일부 레이어만 full attention)을 설계할 때 비용-품질 트레이드오프를 판단하는 기준이 됩니다.

---

## Paper 3 (Recent): Learning to (Learn at Test Time): RNNs with Expressive Hidden States
- **Authors:** Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin
- **Year:** 2024 (v4 revised 2025)
- **arXiv:** https://arxiv.org/abs/2407.04620
- **PDF:** [./ttt-learning-at-test-time-rnns-sun-2024.pdf](./ttt-learning-at-test-time-rnns-sun-2024.pdf)
- **Citation Count:** ~300+

### 요약
RNN의 은닉 상태 자체를 "작은 머신러닝 모델"로 보고, 토큰이 들어올 때마다 자기지도(self-supervised) 손실에 대한 경사하강 한 스텝으로 상태를 갱신하는 TTT(Test-Time Training) 레이어를 제안합니다. 은닉 상태가 선형 모델인 TTT-Linear와 2층 MLP인 TTT-MLP를 제시하며, 125M~1.3B 규모에서 Transformer·Mamba와 대등하거나 우세하고, Mamba와 달리 16k 이상 컨텍스트에서도 perplexity가 계속 내려갑니다.

### 핵심 기여
- "은닉 상태 = 모델, 업데이트 규칙 = 학습"이라는 새로운 RNN 설계 프레임
- 미니배치 TTT와 dual form으로 GPU 친화적 병렬 학습 구현
- 긴 컨텍스트에서 Transformer처럼 스케일하면서 선형 복잡도 유지 (8k에서 Transformer보다 빠름)

### 이 논문이 중요한 이유
Paper 2의 linear attention이 "상태에 정보를 더하는" 방식이라면, TTT는 "상태를 학습시키는" 방식으로 표현력을 확장합니다. 이 아이디어는 Titans, Gated DeltaNet, ATLAS 등 2025년 "테스트 타임 메모리" 계열 연구로 이어졌고, 에이전트의 장기 기억 설계와도 개념적으로 맞닿아 있습니다.

### 사전 지식
Linear attention/RNN 등가성(Paper 2), 경사하강과 자기지도 학습, Mamba 기본 개념

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Titans: Learning to Memorize at Test Time (Behrouz et al., 2025)](https://arxiv.org/abs/2501.00663)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (Yang et al., 2024)](https://arxiv.org/abs/2412.06464)

### 실무 적용
긴 문서·긴 영상·장시간 음성처럼 컨텍스트가 계속 늘어나는 워크로드에서, KV cache 없이 컨텍스트를 "압축 학습"하는 아키텍처 후보입니다. 실제로 TTT 레이어는 1분 길이 영상 생성(One-Minute Video Generation with TTT, 2025)에 적용되었습니다. 아직 프로덕션 표준은 아니므로 PoC·연구 트랙에서 검토하는 것이 현실적입니다.

---

## 추천 읽기 순서
1. **Karpathy et al. (2015)** — RNN 은닉 상태가 실제로 무엇을 저장하는지 직관 확보
2. **Katharopoulos et al. (2020)** — Attention을 "행렬 상태 RNN"으로 다시 보는 관점 획득
3. **Sun et al. (2024)** — 그 상태를 "학습하는 모델"로 확장하는 최신 아이디어

## 핵심 테이크어웨이
- **질문 1: RNN의 상태에는 무엇이 들어있나?** → 해석 가능한 구조적 정보(괄호, 따옴표, 위치)가 개별 셀에 저장된다 (Paper 1).
- **질문 2: Transformer는 왜 RNN보다 비싼가?** → softmax 때문. 커널로 바꾸면 Transformer도 고정 크기 상태의 RNN이 된다 (Paper 2).
- **질문 3: 고정 크기 상태의 한계는 어떻게 넘나?** → 상태를 "더하기"가 아니라 "학습"으로 갱신해 표현력을 키운다 (Paper 3).
- 가설: 현대 시퀀스 모델링의 핵심 축은 "상태 크기 × 상태 갱신 규칙의 표현력"이며, 아키텍처 선택은 결국 메모리 비용과 장기 기억 품질의 트레이드오프다.

## 다음 토픽과의 연결
다음 토픽 **Optimization and Regularization**에서는 Adam/AdamW 등 옵티마이저를 다룹니다. TTT는 "추론 중에 내부에서 경사하강을 돌리는" 모델이므로, 옵티마이저의 학습률·모멘텀·weight decay가 단순한 학습 설정이 아니라 모델 구조의 일부가 될 수 있다는 관점으로 연결해 읽어보세요.
