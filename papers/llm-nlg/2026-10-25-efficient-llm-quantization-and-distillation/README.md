# Daily AI Paper Recommendations

> **Date:** 2026-10-25
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** Efficient LLM Quantization and Distillation

> 이번 회차는 이전 사이클에서 다룬 양자화/LoRA 계열과 겹치지 않도록 **프루닝(Pruning) + 지식 증류(Distillation)** 축에 초점을 맞췄습니다.

---

## Paper 1 (Classic): SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot
- **Authors:** Elias Frantar, Dan Alistarh
- **Year:** 2023 (ICML 2023)
- **arXiv:** https://arxiv.org/abs/2301.00774
- **PDF:** [./sparsegpt-frantar-2023.pdf](./sparsegpt-frantar-2023.pdf)
- **Citation Count:** ~900+

### 요약
GPT 계열 대형 모델(OPT-175B, BLOOM-176B)을 재학습 없이 한 번에(one-shot) 50~60% 희소화해도 perplexity 손실이 거의 없음을 보인 논문입니다. 레이어별로 가중치를 제거하면서 남은 가중치를 근사 2차 정보(Hessian)로 보정하는 효율적 솔버를 제안해, 1,750억 파라미터 모델을 단일 GPU에서 4.5시간 안에 프루닝합니다.

### 핵심 기여
- 대규모 레이어별 희소 회귀 문제를 Hessian 역행렬 공유로 빠르게 푸는 근사 재구성 알고리즘
- 비정형 희소성뿐 아니라 하드웨어 친화적인 2:4 / 4:8 반정형(semi-structured) 희소성 지원
- 양자화(GPTQ)와 결합 가능 — 희소화 + 4bit 양자화를 동시에 적용하는 압축 파이프라인 제시

### 이 논문이 중요한 이유
"모델이 클수록 오히려 프루닝에 강하다"는 관찰을 처음 체계적으로 보여줬습니다. GPTQ와 같은 저자·같은 수학적 뼈대(OBS 계열)를 공유하므로, 양자화와 프루닝을 하나의 "레이어별 재구성 문제"로 이해하게 해 주는 핵심 논문입니다.

### 사전 지식
- 2차 테일러 근사, Hessian 개념 (Optimal Brain Surgeon)
- 레이어별 post-training 압축 개념 (GPTQ를 먼저 읽으면 훨씬 쉬움)
- 비정형 vs 정형 희소성, NVIDIA Ampere의 2:4 sparsity

### 관련 논문
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar et al., 2022)](https://arxiv.org/abs/2210.17323)
- [A Simple and Effective Pruning Approach for LLMs / Wanda (Sun et al., 2023)](https://arxiv.org/abs/2306.11695)
- [LLM-Pruner: On the Structural Pruning of Large Language Models (Ma et al., 2023)](https://arxiv.org/abs/2305.11627)

### 실무 적용
- 2:4 희소성 + TensorRT-LLM으로 추론 처리량 향상 (Ampere/Hopper GPU)
- 학습 데이터·재학습 예산이 없는 상황에서 오픈소스 모델을 빠르게 경량화하는 베이스라인
- 프루닝 이후 소량 증류/파인튜닝으로 품질 회복하는 파이프라인의 1단계

---

## Paper 2 (Classic): TinyBERT: Distilling BERT for Natural Language Understanding
- **Authors:** Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, Qun Liu
- **Year:** 2019 (Findings of EMNLP 2020)
- **arXiv:** https://arxiv.org/abs/1909.10351
- **PDF:** [./tinybert-jiao-2019.pdf](./tinybert-jiao-2019.pdf)
- **Citation Count:** ~2,000+

### 요약
출력 logit만 흉내 내는 기존 증류를 넘어, 임베딩·히든 상태·어텐션 행렬까지 레이어 단위로 정렬하는 Transformer 전용 증류 기법을 제안합니다. 사전학습 단계와 태스크 파인튜닝 단계 모두에서 증류하는 2단계 프레임워크로, 4레이어 TinyBERT가 BERT-Base 성능의 96.8%를 유지하면서 7.5배 작고 9.4배 빠릅니다.

### 핵심 기여
- Transformer 증류 손실: 어텐션 기반 + 히든 상태 기반 + 임베딩 + 예측층 증류
- General distillation(사전학습) + Task-specific distillation(파인튜닝)의 2단계 학습
- 데이터 증강을 활용한 태스크 증류 데이터 확장

### 이 논문이 중요한 이유
"선생 모델의 중간 표현까지 전달한다"는 feature-level 증류의 대표 사례로, 이후 MiniLM, 그리고 오늘의 Minitron(로짓 + 중간 상태 증류)까지 이어지는 설계 원칙의 원형입니다. LLM 증류를 공부할 때 '무엇을 증류할 것인가'의 선택지를 정리해 줍니다.

### 사전 지식
- Hinton의 Knowledge Distillation (soft target, temperature)
- BERT 구조와 사전학습/파인튜닝 패러다임
- 어텐션 행렬, MSE/KL 손실

### 관련 논문
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [DistilBERT, a distilled version of BERT (Sanh et al., 2019)](https://arxiv.org/abs/1910.01108)
- [MiniLM: Deep Self-Attention Distillation (Wang et al., 2020)](https://arxiv.org/abs/2002.10957)

### 실무 적용
- 검색·분류·의도 인식 등 온디바이스/저지연 NLU 서비스의 경량 인코더
- RAG의 임베딩/리랭커 모델 경량화
- 대형 LLM을 선생으로 두고 도메인 특화 소형 모델을 만드는 증류 레시피 설계의 출발점

---

## Paper 3 (Recent): Compact Language Models via Pruning and Knowledge Distillation (Minitron)
- **Authors:** Saurav Muralidharan, Sharath Turuvekere Sreenivas, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, Pavlo Molchanov
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** https://arxiv.org/abs/2407.14679
- **PDF:** [./minitron-muralidharan-2024.pdf](./minitron-muralidharan-2024.pdf)
- **Citation Count:** ~150+

### 요약
NVIDIA가 Nemotron-4 15B를 깊이·너비·어텐션 헤드·MLP 축으로 구조적 프루닝한 뒤, 원본 학습 데이터의 3% 미만으로 증류 재학습해 8B/4B 모델(Minitron)을 만든 연구입니다. 각 크기를 처음부터 학습하는 것 대비 최대 40배 적은 토큰으로 MMLU 최대 16% 향상을 달성했고, 모델 패밀리 전체 기준 약 1.8배 연산을 절감했습니다.

### 핵심 기여
- 활성값 기반 중요도 추정으로 4개 축(depth/width/head/MLP)의 구조적 프루닝 실증 비교
- "너비 프루닝이 깊이 프루닝보다 낫다", "재학습은 로짓 증류가 일반 학습보다 낫다" 등 실무 베스트 프랙티스 정리
- 큰 모델 1개 → 작은 모델 패밀리를 저비용으로 파생하는 학습 전략 (가중치 오픈소스 공개)

### 이 논문이 중요한 이유
SparseGPT(무학습 프루닝)와 TinyBERT(중간층 증류)를 LLM 스케일에서 결합한 현대적 레시피입니다. Llama-3.1-Minitron, Mistral-NeMo-Minitron 등 실제 제품 모델로 이어졌기 때문에 "소형 모델을 어떻게 만들 것인가"에 대한 업계 표준 답안에 가깝습니다.

### 사전 지식
- 구조적 vs 비구조적 프루닝 차이 (Paper 1)
- 로짓/중간 상태 증류 (Paper 2)
- 스케일링 법칙과 학습 토큰 수-성능 관계

### 관련 논문
- [LLM Pruning and Distillation in Practice: The Minitron Approach (Sreenivas et al., 2024)](https://arxiv.org/abs/2408.11796)
- [Sheared LLaMA: Accelerating LM Pre-training via Structured Pruning (Xia et al., 2023)](https://arxiv.org/abs/2310.06694)
- [Gemma 2: Improving Open Language Models at a Practical Size (Gemma Team, 2024)](https://arxiv.org/abs/2408.00118)

### 실무 적용
- 자사 도메인 모델(예: 더빙 스크립트 번역/요약용 LLM)을 대형 → 소형으로 파생해 서빙 비용 절감
- 엣지·온디바이스 에이전트용 4B급 모델 확보 전략
- 프루닝 + 증류 + 양자화(AWQ/GPTQ)를 직렬로 묶은 압축 파이프라인 설계

---

## 추천 읽기 순서
1. **TinyBERT** — "무엇을 증류하는가"(로짓 vs 중간 표현) 개념을 먼저 잡기
2. **SparseGPT** — 재학습 없는 프루닝의 수학적 원리(레이어별 재구성) 이해
3. **Minitron** — 두 아이디어를 LLM 규모에서 결합한 실전 레시피로 마무리

## 핵심 테이크어웨이
- **Q. 압축의 세 축은?** 양자화(비트 수 줄이기), 프루닝(파라미터 수 줄이기), 증류(작은 모델에 지식 옮기기) — 서로 경쟁이 아니라 **조합**하는 관계입니다.
- **Q. 재학습 없이 얼마나 줄일 수 있나?** SparseGPT는 대형 모델에서 50% 희소성까지 거의 무손실. 단, 구조적으로 크게 줄이려면 증류 재학습이 필요합니다.
- **Q. 소형 모델을 처음부터 학습할까, 큰 모델에서 깎아낼까?** Minitron의 답은 "깎고 증류하라" — 토큰 40배 절감.
- **Q. 증류 신호로 무엇을 쓰나?** 로짓 KL이 기본, 중간 히든 상태 정렬은 추가 이득(TinyBERT → Minitron).

## 다음 토픽과의 연결
다음 모듈은 **Chain-of-Thought와 Few-Shot Prompting**입니다. 압축된 소형 모델은 추론 능력이 약해지기 쉬운데, CoT 프롬프팅과 "추론 과정 증류(CoT distillation)"가 이를 보완하는 핵심 수단입니다. 오늘 배운 증류 개념이 다음 주제에서 "큰 모델의 추론 과정을 작은 모델에 옮기기"로 다시 등장합니다.
