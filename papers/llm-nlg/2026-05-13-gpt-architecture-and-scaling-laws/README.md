# Daily AI Paper Recommendations

> **Date:** 2026-05-13
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

---

## Paper 1 (Classic): Training Compute-Optimal Large Language Models (Chinchilla)
- **Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, et al. (DeepMind)
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2203.15556
- **PDF:** [./chinchilla-hoffmann-2022.pdf](./chinchilla-hoffmann-2022.pdf)
- **Citation Count:** 약 4,500+회 (2026년 기준)

### 요약
이 논문은 기존 Kaplan et al.(2020) 스케일링 법칙이 모델 크기를 과대평가하고 데이터를 과소평가했음을 실증적으로 보여줍니다. 동일한 컴퓨트 예산에서는 모델 파라미터 수와 학습 토큰 수를 거의 1:1 비율로 함께 키워야 최적이라는 것을 400개 이상의 모델 실험으로 입증했고, 이 가설을 바탕으로 학습한 70B 파라미터 Chinchilla 모델이 280B Gopher를 거의 모든 벤치마크에서 능가했습니다.


### 핵심 기여
- 컴퓨트-최적(compute-optimal) 학습 시 파라미터(N)와 토큰(D)이 거의 같은 비율로 스케일링되어야 함을 IsoFLOP, 파라메트릭 손실 모델, 작은 모델 외삽 세 가지 접근으로 일관되게 증명
- Kaplan(2020)이 제시한 N^0.73 vs D^0.27 비율을 N^≈0.5, D^≈0.5로 수정 — 즉 "데이터가 부족하다(data-starved)"는 새로운 패러다임 제시
- Chinchilla(70B/1.4T tokens)가 Gopher(280B), GPT-3(175B), Megatron-Turing NLG(530B)를 MMLU 포함 다수 벤치마크에서 능가하며 가설 검증

### 이 논문이 중요한 이유
AI 엔지니어 입장에서 Chinchilla는 "큰 모델 = 더 좋은 모델"이라는 통념을 무너뜨리고, 동일한 GPU/TPU 예산으로 어떻게 더 강한 모델을 학습할지에 대한 의사결정 프레임을 제공합니다. 이후 LLaMA, Mistral, Qwen, DeepSeek 등 거의 모든 오픈소스 LLM이 Chinchilla 비율(약 20 tokens/param) 또는 그 이상으로 토큰을 확장하는 전략을 채택했고, 추론 효율을 위해 의도적으로 "over-trained" 하는 시대(Llama 3는 15T tokens/8B 모델 = 1,875 tokens/param)를 열었습니다.

### 사전 지식
Transformer 아키텍처와 cross-entropy loss 기반 언어모델 학습, Kaplan et al.(2020) 원조 스케일링 법칙(N^0.73), FLOPs 계산식(approx. 6ND), AdamW 옵티마이저 및 cosine learning rate scheduler에 대한 기본 이해가 필요합니다.

### 관련 논문
- [Scaling Laws for Neural Language Models (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361)
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [Scaling Language Models: Methods, Analysis & Insights from Gopher (Rae et al., 2021)](https://arxiv.org/abs/2112.11446)

### 실무 적용
사전학습 단계에서 GPU-hour 예산이 정해졌을 때 "어느 크기의 모델을 몇 토큰으로 학습할 것인가"를 결정하는 표준 도구입니다. 또한 모델 서빙 비용을 줄이기 위해 Chinchilla 비율을 넘어 over-training(예: Llama 3, Qwen 2.5)을 하는 것이 추론 효율 측면에서 합리적임을 정량적으로 설명해주며, 사내 LLM 학습 계획 수립 시 IsoFLOP 곡선을 그려 모델 크기와 데이터 크기의 trade-off를 시각화하는 데 사용됩니다.

---

## Paper 2 (Classic): LLaMA - Open and Efficient Foundation Language Models
- **Authors:** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, et al. (Meta AI)
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2302.13971
- **PDF:** [./llama-touvron-2023.pdf](./llama-touvron-2023.pdf)
- **Citation Count:** 약 12,000+회 (2026년 기준)

### 요약
LLaMA는 7B부터 65B 파라미터까지의 공개 데이터만으로 학습된 foundation LLM 시리즈로, 13B 모델이 GPT-3(175B)를 대부분의 벤치마크에서 능가하고 65B가 PaLM-540B나 Chinchilla-70B와 경쟁할 수 있음을 보여주었습니다. 추론 효율을 위해 작은 모델을 더 많은 토큰(1.0~1.4T tokens)으로 over-train하는 전략을 채택했고, 가중치를 학계에 공개하면서 이후 오픈소스 LLM 생태계의 출발점이 되었습니다.


### 핵심 기여
- 추론 비용을 핵심 KPI로 두고, Chinchilla-optimal보다 토큰을 더 많이 학습(over-training)하여 작은 모델로도 거대 모델 수준의 성능을 달성
- RMSNorm(Pre-normalization), SwiGLU 활성화, Rotary Positional Embedding(RoPE) 등 후속 LLM의 사실상 표준이 된 아키텍처 변경을 한 논문에 종합
- 공개 데이터(CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, StackExchange)만 사용했음에도 GPT-3 수준의 성능을 재현해, "공개 데이터 + 효율적 아키텍처"가 가능하다는 것을 증명

### 이 논문이 중요한 이유
LLaMA는 오픈소스 LLM 생태계의 빅뱅이었습니다. 가중치가 공개되면서 Alpaca, Vicuna, Llama 2, Mistral, Qwen, Code Llama, Llama 3로 이어지는 패밀리 및 파인튜닝 워크플로우(LoRA/QLoRA)가 폭발적으로 발전했고, AI 엔지니어 입장에서 LLaMA 아키텍처(RMSNorm + SwiGLU + RoPE)는 사실상 현대 디코더 LLM의 "표준 템플릿"이 되었습니다. 사내 LLM/SLM을 설계할 때 출발점이 되는 reference 아키텍처입니다.

### 사전 지식
Transformer 디코더, LayerNorm vs RMSNorm 차이, GeLU/Swish/SwiGLU 활성화 함수, absolute vs rotary positional embedding, AdamW optimizer, FSDP/Tensor Parallelism 같은 분산 학습 기법에 대한 이해가 필요합니다.

### 관련 논문
- [Training Compute-Optimal Large Language Models / Chinchilla (Hoffmann et al., 2022)](https://arxiv.org/abs/2203.15556)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer / SwiGLU (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models (Touvron et al., 2023)](https://arxiv.org/abs/2307.09288)

### 실무 적용
B2B/B2C SaaS에서 자체 LLM을 운영해야 할 때 LLaMA 계열 아키텍처는 vLLM, TGI, Llama.cpp, Ollama, MLC 등 거의 모든 추론 엔진에서 최적화가 되어 있어 운영 효율이 가장 좋습니다. 또한 도메인 특화(예: 의료, 금융, 음성 AI)에서 7B~13B 모델을 도메인 데이터로 continued pre-training하거나 LoRA로 파인튜닝하는 표준 레시피가 LLaMA 기반으로 정립되어 있어, AI Dubbing/Avatar 같은 서비스에 적합한 SLM(Small Language Model)을 구축할 때 직접 적용 가능합니다.

---

## Paper 3 (Recent): DeepSeek-V3 Technical Report
- **Authors:** DeepSeek-AI Team (Aixin Liu, Bei Feng, Bing Xue, et al.)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2412.19437
- **PDF:** [./deepseek-v3-technical-report-deepseek-2024.pdf](./deepseek-v3-technical-report-deepseek-2024.pdf)
- **Citation Count:** 약 2,500+회 (2026년 기준, 빠르게 증가 중)

### 요약
DeepSeek-V3는 총 671B 파라미터 중 토큰당 37B만 활성화되는 Mixture-of-Experts(MoE) 디코더 LLM으로, 14.8T 토큰을 약 2.788M H800 GPU-시간(약 $5.6M)에 학습해 GPT-4o, Claude 3.5 Sonnet 수준의 성능을 오픈웨이트로 달성했습니다. Multi-head Latent Attention(MLA), DeepSeekMoE, auxiliary-loss-free load balancing, FP8 mixed-precision training, Multi-Token Prediction(MTP) 등 학습/추론 효율을 위한 새로운 아키텍처 및 시스템 기법을 종합적으로 도입한 것이 특징입니다.


### 핵심 기여
- **MLA(Multi-head Latent Attention):** KV 캐시를 low-rank latent vector로 압축해 추론 메모리/속도를 크게 개선
- **DeepSeekMoE + auxiliary-loss-free balancing:** 256개 routed expert + 1개 shared expert 구조에 보조 손실 없이 라우팅을 안정화해 토큰당 활성 파라미터를 37B로 유지하면서 expert specialization 극대화
- **FP8 학습 + DualPipe pipeline parallelism:** 대규모 MoE를 FP8로 안정적으로 학습하는 최초의 공개 시도로 학습 비용을 약 $5.6M까지 절감하여 "효율의 새 기준선" 제시
- **Multi-Token Prediction(MTP) 학습 목표**로 데이터 효율과 speculative decoding 호환성을 동시에 확보

### 이 논문이 중요한 이유
2024–2025 LLM 트렌드의 핵심은 "고정 컴퓨트 예산 안에서 어떻게 더 똑똑하게 만들 것인가" 즉 efficient scaling으로 이동했습니다. DeepSeek-V3는 Chinchilla 이후 스케일링 법칙의 후속 답안 — MoE + 효율적인 attention + 저정밀(FP8) — 을 한 보고서에 종합했고, 폐쇄형 frontier 모델과의 격차를 1/10 비용으로 줄일 수 있음을 보여주어 AI 엔지니어가 알아야 할 "현대적 LLM 학습 레퍼런스"가 되었습니다. 이후 Llama 4, Qwen 3, Kimi 등이 DeepSeek-V3 패턴(MoE + MLA-like attention + FP8)을 채택하는 흐름을 만들었습니다.

### 사전 지식
LLaMA 아키텍처(RMSNorm/SwiGLU/RoPE), Sparse MoE(Switch Transformer, Mixtral), KV cache와 GQA/MQA, FP16/BF16/FP8 mixed-precision training, pipeline parallelism의 bubble 문제, speculative decoding의 기본 개념이 있어야 합니다.

### 관련 논문
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model (DeepSeek-AI, 2024)](https://arxiv.org/abs/2405.04434)
- [The Llama 3 Herd of Models (Dubey et al., 2024)](https://arxiv.org/abs/2407.21783)

### 실무 적용
Agentic AI / B2B SaaS에서 추론 비용은 곧 ARPU와 마진을 좌우합니다. DeepSeek-V3가 보여준 MoE + MLA 패턴은 동일 품질 대비 토큰 비용을 5–10배 절감할 수 있어, 자체 호스팅 시나리오에서 ROI가 매우 큽니다. 또한 FP8 학습 + DualPipe는 GPU 비용이 부담스러운 스타트업/CPO 입장에서 "1,000개 미만 GPU로도 frontier급 모델을 학습할 수 있다"는 가능성을 제시하며, AI 네이티브 제품의 데이터 플라이휠을 자체 모델로 돌리는 의사결정의 근거가 됩니다.

---

## 추천 읽기 순서
1. **Chinchilla (2022)** — "데이터가 부족하다"는 스케일링의 본질을 먼저 잡고
2. **LLaMA (2023)** — 그 위에 만들어진 사실상의 모던 디코더 LLM 표준 아키텍처를 익히고
3. **DeepSeek-V3 (2024)** — 효율 스케일링(MoE + MLA + FP8)이 어디까지 왔는지 최신 답안을 확인

이 순서로 읽으면 "왜 모델 크기보다 데이터/구조/정밀도 효율이 중요해졌는가"라는 흐름이 자연스럽게 연결됩니다.

## 핵심 테이크어웨이
- **컴퓨트-최적은 곧 데이터-중심:** 동일 FLOPs에서 N과 D는 거의 같은 비율로 키워야 한다(Chinchilla). 하지만 추론 비용이 더 중요한 실무에서는 의도적인 over-training이 합리적이다(LLaMA, Llama 3).
- **표준 아키텍처는 RMSNorm + SwiGLU + RoPE:** 이후 모든 오픈소스 LLM의 baseline. 여기서 출발해 GQA/MLA, MoE, MTP 같은 효율 기법을 추가하는 것이 현대적인 디자인 패턴이다.
- **효율의 새 frontier는 MoE × MLA × FP8:** DeepSeek-V3가 보여준 것처럼 frontier 성능을 1/10 비용으로 재현하는 것이 가능해졌고, 자체 LLM을 운영하려는 회사에 결정적 기회가 열렸다.
- **스케일링 법칙은 도구이지 진리가 아니다:** Kaplan → Chinchilla → over-training → MoE로 매 2년마다 "최적"의 정의가 바뀐다. 항상 자기 워크로드의 cost/latency/quality 함수에 맞춰 다시 풀어야 한다.

## 다음 토픽과의 연결
다음 토픽인 **"Instruction Tuning and RLHF"**는 이번에 다룬 사전학습된 base LLM을 사람이 원하는 방향으로 정렬하는 단계입니다. Chinchilla/LLaMA/DeepSeek-V3로 만든 강력한 base 모델이 있어야 SFT → RLHF/DPO/GRPO 같은 정렬 기법이 효과를 발휘하며, 특히 DeepSeek-V3가 후속 DeepSeek-R1에서 GRPO 기반 reasoning 학습으로 이어진 흐름을 다음 토픽에서 더 자세히 살펴보게 됩니다.
