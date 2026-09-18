# Daily AI Paper Recommendations

> **Date:** 2026-09-19
> **Module:** Module 4: NLP and Speech Data
> **Topic:** Attention Mechanism and Transformer

---

## Paper 1 (Classic): Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
- **Authors:** Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret
- **Year:** 2020 (ICML 2020)
- **arXiv:** https://arxiv.org/abs/2006.16236
- **PDF:** [./transformers-are-rnns-katharopoulos-2020.pdf](./transformers-are-rnns-katharopoulos-2020.pdf)
- **Citation Count:** ~2,400+

### 요약
소프트맥스 어텐션을 커널 함수의 내적으로 일반화하면, 행렬 곱의 결합법칙을 이용해 O(N²)를 O(N)으로 낮출 수 있다는 것을 보인 논문이다. 나아가 인과적(causal) 마스킹을 적용한 선형 어텐션은 수학적으로 "은닉 상태를 누적하는 RNN"과 동일하다는 사실을 증명했다. 그 결과 자기회귀 생성 시 토큰당 상수 시간·상수 메모리로 추론이 가능해져, 긴 시퀀스에서 최대 4000배의 속도 향상을 보고했다.

### 핵심 기여
- 소프트맥스를 임의의 양수 특징맵 φ(·)로 대체하는 **선형화된 어텐션** 정식화 (QK^T V → Q(K^T V))
- Causal linear attention == RNN 이라는 **등가성 증명**: 어텐션과 순환 신경망을 하나의 프레임으로 통합
- 자기회귀 디코딩을 시퀀스 길이에 대해 **O(1) per-token**으로 만드는 실용적 구현과 대규모 속도 벤치마크

### 이 논문이 중요한 이유
오늘날 Mamba, RWKV, GLA(Gated Linear Attention), RetNet, DeltaNet 같은 "어텐션 대체" 아키텍처 계열의 이론적 출발점이다. AI 엔지니어 입장에서 이 논문은 "왜 Transformer는 KV 캐시가 계속 커지는가", "왜 선형 어텐션 계열은 고정 크기 상태를 갖는가"라는 추론 비용 구조의 근본 원리를 알려준다. LLM 서빙 비용의 대부분이 KV 캐시 메모리 대역폭에서 나온다는 점을 생각하면, 이 논문의 관점은 그대로 비용 구조 설계로 직결된다.

### 사전 지식
- Scaled dot-product attention과 multi-head attention의 수식 (Attention Is All You Need)
- 행렬 곱 결합법칙과 연산량 분석 (N×d 행렬에서 (QK^T)V vs Q(K^T V)의 FLOPs 차이)
- 커널 트릭(kernel method)의 기본 개념과 특징맵(feature map)
- RNN의 은닉 상태 갱신 및 자기회귀 디코딩 과정

### 관련 논문
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Rethinking Attention with Performers (Choromanski et al., 2020)](https://arxiv.org/abs/2009.14794)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Gated Linear Attention Transformers with Hardware-Efficient Training (Yang et al., 2023)](https://arxiv.org/abs/2312.06635)

### 실무 적용
긴 문서 요약, 실시간 음성 스트리밍, 온디바이스 LLM처럼 **메모리 상한이 고정되어야 하는** 제품에서 선형 어텐션/SSM 계열을 채택하는 근거가 된다. 실제로 RWKV·Mamba 기반 모델이 엣지 디바이스 추론에 쓰이는 이유가 여기 있다. 반대로 정밀한 검색(retrieval)이 필요한 태스크에서는 고정 상태가 정보 병목이 되므로, 제품 요구사항(정확도 vs 지연시간/비용)에 따라 하이브리드 선택이 필요하다 — 이 트레이드오프는 Paper 3로 이어진다.

---

## Paper 2 (Classic): GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai
- **Year:** 2023 (EMNLP 2023)
- **arXiv:** https://arxiv.org/abs/2305.13245
- **PDF:** [./gqa-ainslie-2023.pdf](./gqa-ainslie-2023.pdf)
- **Citation Count:** ~1,500+

### 요약
Multi-Query Attention(MQA)은 KV 헤드를 1개로 줄여 디코딩 속도를 크게 높이지만 품질 저하와 학습 불안정을 유발한다. 이 논문은 그 중간 지점인 **Grouped-Query Attention(GQA)** 을 제안한다. 여러 쿼리 헤드가 하나의 KV 헤드를 공유하는 그룹 구조로, MHA에 가까운 품질을 유지하면서 MQA에 가까운 속도를 얻는다. 또한 이미 학습된 MHA 체크포인트를 원래 사전학습 연산량의 5%만으로 GQA/MQA로 **업트레이닝(uptraining)** 하는 방법을 제시했다.

### 핵심 기여
- MHA와 MQA를 양 끝점으로 하는 **연속적인 스펙트럼**으로서의 GQA 정의 (그룹 수 G = H이면 MHA, G = 1이면 MQA)
- KV 헤드를 평균 풀링해 기존 체크포인트를 변환하는 **저비용 업트레이닝 레시피** (α ≈ 5% 추가 학습)
- T5-XXL 기준으로 MQA 수준의 추론 지연시간에서 MHA에 근접한 품질을 유지함을 실증

### 이 논문이 중요한 이유
Llama 2/3, Mistral, Qwen, Gemma 등 현행 오픈 LLM 거의 전부가 GQA를 기본 채택하고 있다. 즉 이 논문은 "연구 아이디어"가 아니라 **오늘 여러분이 서빙하는 모델의 실제 설계**다. KV 캐시 크기가 곧 배치 크기와 최대 컨텍스트 길이를 결정하므로, GQA의 그룹 수는 GPU 한 장에서 동시에 처리 가능한 요청 수(=단가)를 직접 좌우한다. AI 엔지니어가 모델 config의 `num_key_value_heads` 값을 해석하려면 반드시 읽어야 한다.

### 사전 지식
- Multi-Head Attention의 헤드별 Q/K/V 프로젝션 구조
- KV 캐시(KV cache)의 동작 원리와 메모리 크기 계산식 (2 × layers × kv_heads × head_dim × seq_len × batch × dtype)
- 자기회귀 디코딩이 연산 바운드가 아니라 **메모리 대역폭 바운드**라는 점
- [Fast Transformer Decoding / MQA (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)의 기본 아이디어

### 관련 논문
- [Fast Transformer Decoding: One Write-Head is All You Need / MQA (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)
- [Efficient Memory Management for LLM Serving with PagedAttention / vLLM (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model / MLA (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

### 실무 적용
LLM 서빙 단가 최적화의 첫 번째 레버다. GQA 덕분에 동일 GPU에서 배치 크기를 수 배 키울 수 있고, 이는 그대로 토큰당 원가 하락으로 이어진다. vLLM·TensorRT-LLM·SGLang 같은 서빙 엔진의 처리량 계산은 모두 KV 헤드 수를 전제로 한다. 또한 "업트레이닝" 아이디어는 자체 파인튜닝 모델을 처음부터 다시 학습하지 않고 추론 효율만 개선하는 실무적 경로를 제공한다 — 사내 도메인 모델을 보유한 팀이라면 바로 적용 가능한 옵션이다.

---

## Paper 3 (Recent): Hybrid Linear Attention Done Right: Efficient Distillation and Effective Architectures for Extremely Long Contexts
- **Authors:** Yingfa Chen, Zhen Leng Thai, Zihan Zhou, Zhu Zhang, Xingyu Shen, Shuo Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu
- **Year:** 2026 (arXiv preprint, 2026-01-29)
- **arXiv:** https://arxiv.org/abs/2601.22156
- **PDF:** [./hybrid-linear-attention-halo-chen-2026.pdf](./hybrid-linear-attention-halo-chen-2026.pdf)
- **Citation Count:** 신규 논문 (인용 수 집계 초기 단계)

### 요약
소프트맥스 어텐션 블록과 RNN 블록을 섞은 하이브리드 구조는 긴 컨텍스트에서 성능과 처리량의 좋은 균형을 보이지만, 처음부터 사전학습하려면 비용이 너무 크다는 문제가 있었다. 이 논문은 기존 Transformer를 하이브리드로 **증류(distillation)** 하는 파이프라인 HALO(Hybrid Attention via Layer Optimization)와, HyPE라는 새로운 위치 인코딩을 갖춘 하이브리드 아키텍처 HypeNet을 제안한다. Qwen3 계열을 단 2.3B 토큰(원 사전학습 데이터의 0.01% 미만)으로 변환해 원본과 대등한 품질에 더 나은 장문 성능과 효율을 달성했다.

### 핵심 기여
- 어떤 레이어를 RNN 블록으로 바꿀지 최적화하는 **레이어 선택 기반 증류 파이프라인(HALO)**
- 기존 변환 기법이 10B+ 토큰을 요구하던 것을 **2.3B 토큰으로 단축**하고, 동시에 장문 성능 저하 문제를 해결
- 길이 일반화(length generalization)를 크게 개선하는 **HyPE 위치 인코딩**과 아키텍처 수정안, 그리고 Qwen3 시리즈 변환 실증

### 이 논문이 중요한 이유
Paper 1(선형 어텐션의 이론)과 Paper 2(MHA 기반 실전 최적화)가 만나는 지점이다. 업계의 현실적 질문은 "선형 어텐션으로 갈아탈까"가 아니라 "이미 가진 Transformer 체크포인트를 어떻게 싸게 효율화할까"인데, 이 논문은 정확히 그 경로를 보여준다. Paper 2의 업트레이닝 아이디어가 GQA 수준의 국소적 변경이었다면, HALO는 레이어 자체를 순환 구조로 바꾸는 더 공격적인 변환이다. 2026년 현재 장문 컨텍스트 서빙 비용 문제에 대한 가장 실용적인 접근 중 하나다.

### 사전 지식
- Paper 1의 선형 어텐션-RNN 등가성과 고정 크기 상태의 개념
- 지식 증류(knowledge distillation)와 파라미터 전이(parameter transfer)의 기본 절차
- RoPE 등 위치 인코딩과 길이 외삽(length extrapolation) 문제
- 하이브리드 아키텍처 계열(Jamba, Samba, Zamba 등)의 기본 구성

### 관련 논문
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Jamba: A Hybrid Transformer-Mamba Language Model (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)

### 실무 적용
장문 RAG, 코드베이스 전체 분석, 긴 회의록/영상 자막 처리처럼 컨텍스트가 수십만 토큰에 달하는 제품에서 직접적인 원가 절감 수단이 된다. 이미 파인튜닝해둔 모델이 있다면 재사전학습 없이 2~3B 토큰 규모의 변환만으로 장문 처리량을 개선할 수 있다는 점이 핵심이다. 제품 관점에서는 "컨텍스트 길이를 늘리면 단가가 제곱으로 오른다"는 제약이 완화되므로, 기존에 비용 때문에 포기했던 장문 기능의 재검토가 가능해진다.

---

## 추천 읽기 순서

1. **Paper 2 (GQA)** — 가장 실무에 가깝고 수식 부담이 적다. 지금 쓰는 모델의 KV 캐시 구조를 먼저 이해하고 시작하자.
2. **Paper 1 (Transformers are RNNs)** — 왜 KV 캐시가 선형으로 커지는지, 왜 고정 상태 모델이 가능한지의 이론적 근거. 3장의 커널 정식화와 5장의 RNN 등가성 증명만 정독해도 충분하다.
3. **Paper 3 (HALO/HypeNet)** — 앞 두 편의 아이디어가 2026년 실전에서 어떻게 결합되는지 확인. 시간이 없다면 Abstract + 실험 테이블 중심으로.

## 핵심 테이크어웨이

- **어텐션의 비용은 연산이 아니라 메모리에서 온다.** 학습에서는 O(N²) FLOPs가 문제지만, 추론에서는 KV 캐시의 메모리 대역폭이 병목이다. 세 논문 모두 결국 "상태를 얼마나 작게 유지할 것인가"를 다룬다.
- **품질-효율은 이분법이 아니라 스펙트럼이다.** GQA는 MHA↔MQA 사이의 연속체를, 하이브리드 아키텍처는 소프트맥스 어텐션↔RNN 사이의 연속체를 제공한다. 제품 요구사항에 맞춰 이 축 위의 지점을 고르는 것이 엔지니어링 판단이다.
- **처음부터 다시 학습할 필요가 없다.** 업트레이닝(5%)과 증류 변환(0.01%)이라는 두 사례는, 기존 체크포인트 자산을 버리지 않고 효율화하는 경로가 이미 검증되었음을 보여준다.
- **고정 크기 상태에는 대가가 있다.** 무한한 KV 캐시를 상수 크기 상태로 압축하면 정밀 검색(needle-in-a-haystack) 성능이 떨어진다. 하이브리드가 답이 되는 이유이며, 벤치마크 선택 시 반드시 장문 검색 태스크를 포함해야 하는 이유이기도 하다.

## 다음 토픽과의 연결

다음 토픽은 **BERT and Pre-trained Language Models**다. 오늘 다룬 어텐션 변형들은 대부분 디코더(자기회귀 생성) 관점의 효율화였다. 다음 편에서는 같은 Transformer 블록이 양방향 인코더로 쓰일 때 무엇이 달라지는지 — 마스킹 전략, 사전학습 목표(MLM), 그리고 KV 캐시가 없는 인코더에서 효율 병목이 어디로 옮겨가는지를 살펴본다. 특히 GQA 같은 디코딩 최적화가 인코더에서는 왜 무의미한지 비교해보면 두 계열의 구조적 차이가 선명해진다.
