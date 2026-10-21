# Daily AI Paper Recommendations

> **Date:** 2026-10-22
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

> 이번 사이클은 이전 사이클(GPT-1/2/3, Kaplan/Chinchilla, LLaMA, PaLM, Emergent Abilities, Switch Transformers, Mistral 7B)과 겹치지 않도록 **"현대 GPT 아키텍처를 구성하는 부품"** — 위치 인코딩(RoPE)과 어텐션 KV 구조(GQA → MLA) — 에 초점을 맞췄습니다. 오늘의 질문: *"왜 Llama·Qwen·DeepSeek는 모두 원조 GPT와 다른 어텐션을 쓰는가?"*

---

## Paper 1 (Classic): RoFormer: Enhanced Transformer with Rotary Position Embedding
- **Authors:** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2104.09864
- **PDF:** [./roformer-rope-su-2021.pdf](./roformer-rope-su-2021.pdf)
- **Citation Count:** ~3,000+

### 요약
쿼리·키 벡터를 위치에 비례하는 각도만큼 "회전"시키는 Rotary Position Embedding(RoPE)을 제안합니다. 회전 덕분에 두 토큰의 내적이 절대 위치가 아닌 **상대 거리**에만 의존하게 되어, 절대 위치 임베딩의 단순함과 상대 위치 인코딩의 장점을 동시에 얻습니다.

### 핵심 기여
- 절대 위치를 회전 행렬로 인코딩해 어텐션 점수에 상대 위치 정보가 자연스럽게 나타나도록 하는 수식적 설계
- 거리가 멀어질수록 내적이 감쇠하는 성질(long-term decay)을 이론적으로 제시
- 추가 학습 파라미터 없이 선형 어텐션 등 다양한 어텐션 변형과 결합 가능

### 이 논문이 중요한 이유
GPT-NeoX, PaLM, LLaMA 계열, Qwen, Mistral, DeepSeek 등 사실상 모든 현대 오픈 LLM이 RoPE를 씁니다. 또한 컨텍스트 윈도우 확장 기법(Position Interpolation, NTK-aware scaling, YaRN)이 모두 RoPE의 회전 주파수를 조작하는 방식이므로, "128K 컨텍스트"가 어떻게 가능한지 이해하려면 반드시 거쳐야 하는 논문입니다.

### 사전 지식
Transformer의 Self-Attention(Q·K 내적), Sinusoidal 위치 인코딩(Attention Is All You Need), 복소수의 회전(오일러 공식)에 대한 기초 이해

### 관련 논문
- [Self-Attention with Relative Position Representations (Shaw et al., 2018)](https://arxiv.org/abs/1803.02155)
- [Extending Context Window of LLMs via Positional Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension of LLMs (Peng et al., 2023)](https://arxiv.org/abs/2309.00071)

### 실무 적용
오픈 모델을 긴 문서 요약·RAG에 쓰려면 `rope_scaling`(linear/dynamic/yarn) 설정을 조정하게 되는데, 이 값의 의미가 바로 RoPE 주파수 스케일입니다. 파인튜닝 시 학습 길이보다 긴 입력에서 품질이 급락하는 현상도 RoPE 외삽 한계로 설명할 수 있습니다.

---

## Paper 2 (Classic): GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- **Authors:** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai
- **Year:** 2023 (EMNLP 2023)
- **arXiv:** https://arxiv.org/abs/2305.13245
- **PDF:** [./gqa-grouped-query-attention-ainslie-2023.pdf](./gqa-grouped-query-attention-ainslie-2023.pdf)
- **Citation Count:** ~1,500+

### 요약
모든 헤드가 K/V를 따로 갖는 Multi-Head Attention(MHA)과, 하나의 K/V를 공유하는 Multi-Query Attention(MQA)의 중간인 **Grouped-Query Attention(GQA)** 을 제안합니다. 쿼리 헤드를 그룹으로 묶어 그룹별로 K/V를 공유하며, 기존 MHA 체크포인트를 원래 사전학습 연산량의 약 5%만으로 "업트레이닝"해 변환하는 방법도 제시합니다.

### 핵심 기여
- MHA 품질에 근접하면서 MQA 수준의 디코딩 속도를 내는 GQA 구조
- 기존 MHA 체크포인트의 K/V 헤드를 mean-pooling해 MQA/GQA로 변환하는 업트레이닝 레시피
- KV 헤드 수가 품질·속도 트레이드오프를 조절하는 "다이얼"임을 실험으로 보여줌

### 이 논문이 중요한 이유
LLM 추론 비용의 병목은 연산이 아니라 **KV 캐시 메모리 대역폭**입니다. Llama 2 70B, Llama 3 전 모델, Mistral, Qwen2 이후 대부분의 모델이 GQA를 채택했습니다. "모델 아키텍처 결정 = 서빙 비용 결정"이라는 관점을 체득하게 해주는 짧고 실용적인 논문입니다.

### 사전 지식
Multi-Head Attention 구조, 자기회귀 디코딩에서 KV 캐시가 필요한 이유, 메모리 대역폭 바운드 개념

### 관련 논문
- [Fast Transformer Decoding: One Write-Head is All You Need / MQA (Shazeer, 2019)](https://arxiv.org/abs/1911.02150)
- [Efficient Memory Management for LLM Serving with PagedAttention / vLLM (Kwon et al., 2023)](https://arxiv.org/abs/2309.06180)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288)

### 실무 적용
자체 서빙 시 동시 접속자 수와 최대 컨텍스트 길이는 `KV 캐시 = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes`로 결정됩니다. 모델 선택 단계에서 `num_key_value_heads` 값만 봐도 GPU 한 장에 얼마나 많은 세션을 올릴 수 있는지 추정할 수 있어, 에이전트처럼 긴 컨텍스트를 쓰는 서비스의 단가 계산에 직결됩니다.

---

## Paper 3 (Recent): DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- **Authors:** DeepSeek-AI (Aixin Liu, Bei Feng, Bin Wang, et al.)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2405.04434
- **PDF:** [./deepseek-v2-moe-mla-deepseek-ai-2024.pdf](./deepseek-v2-moe-mla-deepseek-ai-2024.pdf)
- **Citation Count:** ~1,000+

### 요약
총 236B 파라미터 중 토큰당 21B만 활성화하는 MoE 모델로, 128K 컨텍스트를 지원합니다. 핵심은 K/V를 저차원 잠재 벡터로 압축해 캐시하는 **Multi-head Latent Attention(MLA)** 과 세분화된 전문가 + 공유 전문가 구조인 **DeepSeekMoE** 이며, 논문은 DeepSeek 67B 대비 학습 비용 42.5% 절감, KV 캐시 93.3% 감소, 최대 생성 처리량 5.76배를 보고합니다.

### 핵심 기여
- MLA: K/V를 low-rank 잠재 벡터로 압축 저장 → GQA보다 적은 캐시로 MHA 수준 이상의 품질을 목표
- Decoupled RoPE: RoPE가 압축 행렬 흡수(absorb)를 방해하는 문제를, 위치 정보를 담당하는 별도 쿼리/키 성분으로 분리해 해결
- DeepSeekMoE(세분화 전문가 + 공유 전문가)와 디바이스 제한 라우팅으로 경제적인 대규모 학습

### 이 논문이 중요한 이유
오늘의 Paper 1(RoPE)과 Paper 2(KV 헤드 공유)가 만나는 지점입니다. GQA가 "K/V를 몇 개 공유할까"였다면 MLA는 "K/V를 압축해서 저장하자"로 한 단계 더 나아가며, 그 과정에서 RoPE와의 충돌을 어떻게 풀었는지가 아키텍처 설계의 교과서적 사례입니다. 이 구조는 DeepSeek-V3/R1로 그대로 이어졌습니다. 단, KV 캐시 감소 수치는 GQA가 아닌 MHA 기반 DeepSeek 67B 대비 결과라는 점에 유의하세요.

### 사전 지식
오늘의 Paper 1·2, Mixture-of-Experts 기본 개념(Switch Transformers), 저차원 행렬 분해(LoRA와 유사한 low-rank 직관)

### 관련 논문
- [DeepSeekMoE: Towards Ultimate Expert Specialization (Dai et al., 2024)](https://arxiv.org/abs/2401.06066)
- [DeepSeek-V3 Technical Report (DeepSeek-AI, 2024)](https://arxiv.org/abs/2412.19437)
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)

### 실무 적용
API 단가가 낮은 모델이 "왜 싼지"를 아키텍처로 설명할 수 있게 됩니다. 자체 SaaS에서 오픈 모델을 고를 때 총 파라미터가 아닌 **활성 파라미터 수(연산 비용)** 와 **토큰당 KV 캐시 크기(메모리 비용)** 를 분리해서 비교하는 습관이 핵심입니다. vLLM·SGLang 등 서빙 엔진의 MLA 지원 여부도 실제 처리량을 크게 좌우합니다.

---

## 추천 읽기 순서
1. **RoFormer (RoPE)** — 위치 정보가 Q·K 내적에 들어가는 방식을 먼저 이해 (3·4장 수식 위주)
2. **GQA** — 6페이지 분량, KV 캐시가 왜 병목인지와 헤드 공유 트레이드오프 파악
3. **DeepSeek-V2** — 2.1절(MLA)과 2.1.3절(Decoupled RoPE)을 앞의 두 논문과 연결하며 정독, 이후 MoE 부분

## 핵심 테이크어웨이
- 현대 GPT는 "Transformer 디코더 + α"이며, 그 α의 상당 부분은 **추론 비용(KV 캐시)을 줄이기 위한 설계**입니다.
- 진화 경로: MHA → MQA → GQA → MLA. 질문은 늘 같습니다 — *"품질을 지키면서 토큰당 저장할 K/V를 얼마나 줄일 수 있는가?"*
- RoPE는 컨텍스트 확장의 기반인 동시에, K/V 압축과 충돌하는 제약이기도 합니다. 부품 간 상호작용을 보는 시각이 아키텍처 이해의 핵심입니다.
- 스케일링 법칙이 "얼마나 크게"를 말한다면, 오늘의 논문들은 "같은 크기를 얼마나 싸게 돌릴까"를 다룹니다. 제품 관점에서는 후자가 단위 경제성을 결정합니다.

## 다음 토픽과의 연결
다음 토픽은 **Instruction Tuning and RLHF**입니다. 오늘 살펴본 사전학습 아키텍처는 "다음 토큰을 잘 예측하는 엔진"까지만 만들어 줍니다. 이 엔진을 사용자의 지시를 따르고 선호에 맞게 답하는 어시스턴트로 바꾸는 과정(InstructGPT, DPO, 그리고 DeepSeek 계열이 사용한 GRPO)을 다음에 다룹니다.
