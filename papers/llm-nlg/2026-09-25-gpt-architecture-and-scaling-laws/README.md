# Daily AI Paper Recommendations

> **Date:** 2026-09-25
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

> 이번 사이클에서는 이전에 다룬 GPT-1/2/3, Kaplan Scaling Laws, Chinchilla, LLaMA, PaLM, Emergent Abilities와 겹치지 않도록 **"파라미터를 늘리되 연산은 늘리지 않는 법"** 이라는 관점에서 논문을 골랐습니다. Sparse 스케일링(MoE) → 작은 Dense 모델의 효율화(Mistral) → 둘을 결합한 최신 오픈 모델 패밀리(Qwen3) 순서입니다.

---

## Paper 1 (Classic): Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
- **Authors:** William Fedus, Barret Zoph, Noam Shazeer
- **Year:** 2021 (JMLR 2022)
- **arXiv:** https://arxiv.org/abs/2101.03961
- **PDF:** [./switch-transformers-fedus-2021.pdf](./switch-transformers-fedus-2021.pdf)
- **Citation Count:** ~3,000+

### 요약
토큰마다 여러 개의 FFN(Expert) 중 **딱 하나만** 골라 쓰는 Top-1 라우팅 Mixture-of-Experts를 제안합니다. 토큰당 연산량(FLOPs)은 그대로 두면서 전체 파라미터 수만 크게 늘릴 수 있어, T5-Base 대비 같은 연산으로 최대 7배 빠른 사전학습 속도를 보였고 1.6조 파라미터 모델까지 확장했습니다.

### 핵심 기여
- 기존 Top-k MoE를 **Top-1(Switch) 라우팅**으로 단순화해 통신·연산 비용과 구현 복잡도를 줄임
- **Load-balancing auxiliary loss**와 **expert capacity factor** 개념으로 특정 Expert에 토큰이 몰리는 문제를 완화
- bfloat16 선택적 정밀도, 작은 초기화 스케일, Expert dropout 등 **희소 모델의 학습 불안정성**을 다루는 실전 기법 정리
- "파라미터 수"가 연산량과 독립적인 **새로운 스케일링 축**이 될 수 있음을 실증

### 이 논문이 중요한 이유
Kaplan/Chinchilla 스케일링 법칙은 "파라미터 ≈ 연산"인 Dense 모델을 전제로 합니다. Switch Transformer는 이 전제를 깨고, 오늘날 DeepSeek-V3, Mixtral, Qwen3-MoE, (추정상) GPT-4 계열까지 이어지는 **MoE 설계의 출발점**이 되었습니다. "왜 235B 모델이 22B만 활성화되는가?"를 이해하려면 반드시 읽어야 합니다.

### 사전 지식
- Transformer의 FFN 블록 구조와 Attention 기초 (Attention Is All You Need)
- Kaplan Scaling Laws의 "파라미터·데이터·연산" 관계
- 데이터/모델 병렬화 개념 (분산 학습에서 all-to-all 통신이 왜 비싼지)

### 관련 논문
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)

### 실무 적용
- MoE 모델을 서빙할 때 **"총 파라미터(메모리)"와 "활성 파라미터(속도·비용)"를 분리**해 GPU 메모리/지연시간을 산정해야 합니다.
- 파인튜닝 시 라우터 붕괴·Expert 불균형이 자주 발생하므로 aux loss, capacity factor 튜닝이 품질에 직결됩니다.
- 제품 관점에서는 "같은 추론 비용으로 더 많은 지식을 담는 모델"을 선택하는 근거가 됩니다.

---

## Paper 2 (Classic): Mistral 7B
- **Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2310.06825
- **PDF:** [./mistral-7b-jiang-2023.pdf](./mistral-7b-jiang-2023.pdf)
- **Citation Count:** ~2,500+

### 요약
7B 파라미터 모델이 모든 벤치마크에서 Llama 2 13B를, 추론·수학·코드에서는 Llama 1 34B를 앞선다는 것을 보여준 짧은 기술 보고서입니다. **Grouped-Query Attention(GQA)** 과 **Sliding Window Attention(SWA)**, Rolling Buffer KV Cache를 결합해 추론 속도와 메모리를 크게 개선했습니다.

### 핵심 기여
- **GQA**로 KV 헤드 수를 줄여 디코딩 속도 향상 및 KV 캐시 메모리 절감
- **SWA(윈도우 4,096)** 를 레이어마다 쌓아, 이론적 수용 범위를 약 131K 토큰까지 확장하면서도 연산은 선형 유지
- **Rolling Buffer Cache + Pre-fill/Chunking**으로 긴 시퀀스 추론 시 캐시 메모리를 고정 크기로 유지
- "작고 잘 만든 모델"이 더 큰 모델을 이길 수 있다는 **추론 효율 중심 설계 철학**을 대중화 (Apache 2.0 공개)

### 이 논문이 중요한 이유
Chinchilla가 "학습 연산 최적"을 말했다면, Mistral 7B는 **"추론 비용 최적"** 관점을 대표합니다. 현업에서 비용을 결정하는 것은 학습이 아니라 수백만 번의 추론이기 때문에, 오버트레이닝된 작은 모델 + 추론 친화적 Attention 설계라는 흐름을 이해하는 데 핵심입니다. 이후 Mixtral(MoE)로 이어지며 Paper 1과도 자연스럽게 연결됩니다.

### 사전 지식
- Multi-Head Attention과 KV Cache가 추론에서 어떻게 동작하는지
- Multi-Query Attention(MQA) 개념
- LLaMA 계열 아키텍처(RMSNorm, SwiGLU, RoPE)

### 관련 논문
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288)
- [Fast Transformer Decoding: One Write-Head is All You Need / MQA (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)

### 실무 적용
- 온디바이스·엣지·저비용 API 서빙에서 **7B급 모델 선택의 기준점**이 되었습니다. vLLM, llama.cpp 등이 GQA/SWA를 기본 지원합니다.
- KV 캐시 크기 = 레이어 × KV 헤드 × head_dim × 토큰 수 → GQA 적용 여부로 동시 사용자 수(throughput)가 크게 달라집니다.
- 도메인 특화 파인튜닝(LoRA/QLoRA)의 베이스 모델로 널리 쓰여, Module 6 Day 17 내용과 바로 이어집니다.

---

## Paper 3 (Recent): Qwen3 Technical Report
- **Authors:** Qwen Team (An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, et al.)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2505.09388
- **PDF:** [./qwen3-technical-report-qwen-2025.pdf](./qwen3-technical-report-qwen-2025.pdf)
- **Citation Count:** ~1,000+ (빠르게 증가 중)

### 요약
0.6B~235B 규모의 Dense 및 MoE 모델(대표: Qwen3-235B-A22B, 활성 22B)로 구성된 오픈 모델 패밀리 보고서입니다. 하나의 모델 안에 **Thinking 모드(단계적 추론)와 Non-thinking 모드(빠른 응답)** 를 통합하고, 추론 토큰 양을 조절하는 **Thinking Budget** 메커니즘을 도입했습니다. 약 36T 토큰, 119개 언어로 사전학습했습니다.

### 핵심 기여
- Dense + MoE 라인업을 동시에 공개하고, MoE에서 **공유 Expert 제거 + global-batch load balancing** 등 설계 선택을 공개
- **Thinking/Non-thinking 통합** 및 Thinking Budget으로 지연시간–품질 트레이드오프를 사용자가 직접 제어
- 대형 모델의 지식을 소형 모델로 옮기는 **Strong-to-Weak Distillation**으로 소형 모델 학습 비용을 크게 절감
- 다국어 지원을 29개 → 119개 언어로 확장 (한국어 포함), Apache 2.0 공개

### 이 논문이 중요한 이유
Paper 1(희소 스케일링)과 Paper 2(추론 효율)가 2025년 프로덕션 모델에서 어떻게 합쳐지는지 보여주는 **종합 사례**입니다. 또한 스케일링의 축이 "학습 연산"에서 **"추론 시 연산(test-time compute)"** 으로 확장되는 흐름을 한 모델의 제품 기능(Thinking Budget)으로 구현했다는 점에서, AI 엔지니어가 최신 아키텍처 트렌드를 한 번에 파악하기 좋습니다.

### 사전 지식
- Paper 1의 MoE 라우팅, Paper 2의 GQA
- RoPE 및 장문 컨텍스트 확장 기법(YaRN 등)
- RLHF/GRPO 등 강화학습 기반 추론 학습 개념 (Day 15와 연결)

### 관련 논문
- [Qwen2.5 Technical Report (Qwen Team, 2024)](https://arxiv.org/abs/2412.15115)
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (Snell et al., 2024)](https://arxiv.org/abs/2408.03314)

### 실무 적용
- 에이전트 서비스에서 **단순 라우팅·분류는 Non-thinking, 복잡한 계획·코드 생성은 Thinking**으로 한 모델 안에서 분기해 비용을 최적화할 수 있습니다.
- Thinking Budget을 제품 설정(예: "빠름/정확" 토글)으로 노출하면 UX와 원가를 동시에 관리할 수 있습니다.
- 한국어 포함 다국어 + Apache 2.0 라이선스라 **자체 호스팅/온프레미스 B2B** 도입 후보로 적합합니다.

---

## 추천 읽기 순서
1. **Mistral 7B** — 9쪽 분량으로 짧고, GQA·SWA·KV 캐시라는 현대 LLM의 추론 최적화 기본기를 먼저 잡습니다.
2. **Switch Transformers** — "연산은 그대로, 파라미터만 늘리기"라는 MoE 스케일링 축을 이해합니다. (Section 2, 3 위주로 읽고 분산 학습 부분은 훑어보기)
3. **Qwen3 Technical Report** — 앞의 두 개념이 최신 프로덕션 모델에서 어떻게 결합되고, test-time compute까지 확장되는지 확인합니다.

## 핵심 테이크어웨이
- **스케일링의 축은 하나가 아니다:** Dense 파라미터(Kaplan) → 데이터 대비 최적 비율(Chinchilla) → 희소 파라미터(MoE) → 추론 시 연산(Thinking) 순으로 확장되어 왔습니다.
- **"총 파라미터"와 "활성 파라미터"를 구분하라:** 메모리 비용과 속도/토큰 비용은 서로 다른 숫자로 결정됩니다.
- **추론 비용이 설계를 이끈다:** GQA, SWA, MoE, Thinking Budget 모두 "서빙 비용을 줄이면서 품질을 지키는" 방향의 선택입니다.
- **질문으로 남겨둘 것:** 우리 제품의 요청 중 몇 %가 Thinking이 필요할까? 활성 파라미터 22B MoE와 32B Dense 중 우리 트래픽 패턴에는 무엇이 더 싸고 좋은가?

## 다음 토픽과의 연결
다음 토픽은 **Instruction Tuning and RLHF**입니다. 오늘 본 모델들은 모두 "사전학습된 베이스 모델"이며, 이를 사람이 원하는 방식으로 대답하게 만드는 것이 후속 학습(SFT, RLHF, DPO, GRPO)입니다. 특히 Qwen3의 Thinking 모드는 RL 기반 추론 학습의 결과물이므로, 내일은 "어떤 학습 신호가 모델을 생각하게 만드는가?"라는 질문을 가지고 InstructGPT와 DPO를 읽어보면 좋습니다.
