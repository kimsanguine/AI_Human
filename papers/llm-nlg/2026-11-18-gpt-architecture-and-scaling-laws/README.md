# Daily AI Paper Recommendations

> **Date:** 2026-11-18
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** GPT Architecture and Scaling Laws

> 이번 사이클 선정 기준: 이전 사이클에서 다룬 GPT-1/2/3, Kaplan Scaling Laws, Chinchilla, PaLM, LLaMA, Switch Transformers, RoPE, GQA, Mistral 7B 등을 제외하고, **"파라미터는 키우되 연산은 아끼는" 아키텍처 계보(MoE + SwiGLU)**에 초점을 맞췄습니다.

---

## Paper 1 (Classic): Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- **Authors:** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean
- **Year:** 2017 (ICLR 2017)
- **arXiv:** https://arxiv.org/abs/1701.06538
- **PDF:** [./sparsely-gated-moe-shazeer-2017.pdf](./sparsely-gated-moe-shazeer-2017.pdf)
- **Citation Count:** ~4,000+

### 요약
수천 개의 "전문가(expert)" 서브네트워크를 두고, 학습 가능한 게이팅 네트워크가 입력 토큰마다 상위 k개의 전문가만 골라 활성화하는 Sparsely-Gated MoE 레이어를 제안했습니다. 이를 통해 연산량(FLOPs)은 거의 그대로 두면서 모델 용량을 1,000배 이상(최대 1,370억 파라미터) 늘렸고, 언어 모델링과 기계번역에서 당시 SOTA를 달성했습니다.

### 핵심 기여
- **조건부 연산(Conditional Computation)의 실용화:** Noisy Top-k Gating으로 토큰마다 일부 전문가만 계산 → "파라미터 수 ≠ 연산량"을 분리
- **로드 밸런싱 손실:** 특정 전문가에만 트래픽이 몰리는 붕괴 현상을 막기 위한 importance/load 보조 손실 도입
- **분산 학습 설계:** 데이터 병렬 + 모델(전문가) 병렬을 섞어 GPU 클러스터에서 거대 MoE를 효율적으로 학습하는 방법 제시

### 이 논문이 중요한 이유
GPT-4(추정), Mixtral, DeepSeek-V3, Qwen3-MoE, Kimi K2 등 현재 프런티어 LLM 상당수가 MoE 구조입니다. Kaplan/Chinchilla 스케일링 법칙이 "더 크게"를 말했다면, 이 논문은 "**어떻게 비용을 감당하며 더 크게 만들 것인가**"에 대한 원조 답안입니다. 라우터, top-k, 로드 밸런싱 손실 같은 용어가 모두 여기서 출발합니다.

### 사전 지식
- LSTM 기반 언어 모델과 기계번역의 기본 구조
- Softmax, 게이팅(gating) 개념
- 데이터 병렬 vs 모델 병렬 학습의 차이

### 관련 논문
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)
- [GShard: Scaling Giant Models with Conditional Computation (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906)

### 실무 적용
MoE 모델을 서빙할 때는 "활성 파라미터"(연산·지연시간 결정)와 "전체 파라미터"(GPU 메모리 결정)를 분리해서 비용을 계산해야 합니다. 예를 들어 에이전트 제품에서 MoE 오픈모델을 자체 호스팅한다면, 처리량은 Dense 소형 모델 수준이지만 VRAM은 대형 모델 수준이 필요하다는 점이 인프라 의사결정의 핵심 변수가 됩니다.

---

## Paper 2 (Classic): GLU Variants Improve Transformer
- **Authors:** Noam Shazeer
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2002.05202
- **PDF:** [./glu-variants-improve-transformer-shazeer-2020.pdf](./glu-variants-improve-transformer-shazeer-2020.pdf)
- **Citation Count:** ~1,500+

### 요약
Transformer의 FFN(Feed-Forward Network) 층에서 ReLU/GELU 대신 Gated Linear Unit(GLU) 계열 활성화(GEGLU, SwiGLU 등)를 사용하면, 동일한 파라미터·연산량에서 사전학습 perplexity와 다운스트림 성능이 일관되게 좋아진다는 것을 보인 4페이지짜리 짧은 논문입니다.

### 핵심 기여
- **SwiGLU / GEGLU 제안:** FFN을 `(xW ⊙ Swish(xV))W₂` 형태의 게이트 구조로 교체
- **공정한 비교 설계:** 행렬이 3개로 늘어나는 만큼 은닉 차원을 2/3로 줄여 파라미터·FLOPs를 맞춘 상태로 비교
- **실증적 개선:** T5 설정에서 GLUE/SuperGLUE/SQuAD 등 다수 벤치마크 성능 향상

### 이 논문이 중요한 이유
PaLM, LLaMA 1~3, Mistral/Mixtral, Qwen, Gemma, DeepSeek 등 거의 모든 현대 LLM이 SwiGLU(또는 GEGLU)를 기본 FFN으로 씁니다. LLaMA 계열의 이상한 FFN 차원(예: 11008)이 왜 `8/3 × d_model`인지 이 논문을 읽으면 바로 이해됩니다. 또한 MoE의 각 "전문가"가 바로 이 SwiGLU FFN이므로 Paper 1·3과 직접 연결됩니다.

### 사전 지식
- Transformer 블록 구조(Attention + FFN + Residual + Norm)
- ReLU, GELU, Swish(SiLU) 활성화 함수
- GLU(Dauphin et al., 2017)의 게이트 개념

### 관련 논문
- [Language Modeling with Gated Convolutional Networks / GLU (Dauphin et al., 2016)](https://arxiv.org/abs/1612.08083)
- [Searching for Activation Functions / Swish (Ramachandran et al., 2017)](https://arxiv.org/abs/1710.05941)
- [PaLM: Scaling Language Modeling with Pathways (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311)

### 실무 적용
LLM을 직접 설계·파인튜닝하거나 LoRA 타깃 모듈을 고를 때 `gate_proj / up_proj / down_proj`라는 이름이 바로 SwiGLU의 세 행렬입니다. 어떤 모듈에 LoRA를 걸지, 양자화 시 어느 층이 민감한지 판단하려면 이 구조를 알아야 합니다.

---

## Paper 3 (Recent): Mixtral of Experts
- **Authors:** Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. (Mistral AI)
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2401.04088
- **PDF:** [./mixtral-of-experts-jiang-2024.pdf](./mixtral-of-experts-jiang-2024.pdf)
- **Citation Count:** ~2,000+

### 요약
Mistral 7B 구조의 각 FFN을 8개의 SwiGLU 전문가로 바꾸고 토큰마다 2개만 활성화하는 Sparse MoE 모델 Mixtral 8x7B를 공개했습니다. 전체 47B 파라미터 중 토큰당 13B만 사용하면서 Llama 2 70B 및 GPT-3.5와 동등하거나 더 나은 성능을 보였고, Apache 2.0으로 공개되어 오픈소스 MoE 시대를 열었습니다.

### 핵심 기여
- **실전형 MoE 레시피:** Top-2 라우팅 + SwiGLU 전문가 + GQA + 32K 컨텍스트를 결합한 디코더 전용 구조
- **비용 대비 성능:** 활성 파라미터 13B로 70B Dense 모델 수준 → 추론 속도·비용 대폭 절감
- **라우팅 분석:** 전문가가 "주제(도메인)"보다 "구문/토큰 형태"에 따라 특화된다는 관찰 → MoE 직관에 대한 통념 수정
- **Mixtral-Instruct:** SFT + DPO로 정렬한 채팅 모델도 함께 공개

### 이 논문이 중요한 이유
2017년 MoE 아이디어(Paper 1)와 2020년 SwiGLU(Paper 2)가 실제 프로덕션급 오픈 LLM으로 어떻게 조립되는지 보여주는 "완성형 사례"입니다. 이후 DeepSeek-MoE/V3, Qwen-MoE, DBRX, Llama 4 등 대부분의 MoE LLM이 Mixtral과의 비교에서 출발합니다.

### 사전 지식
- Paper 1(MoE 라우팅·로드 밸런싱), Paper 2(SwiGLU FFN)
- Mistral 7B 아키텍처(Sliding Window Attention, GQA)
- DPO 기반 정렬의 기본 개념

### 관련 논문
- [Mistral 7B (Jiang et al., 2023)](https://arxiv.org/abs/2310.06825)
- [DeepSeekMoE: Towards Ultimate Expert Specialization (Dai et al., 2024)](https://arxiv.org/abs/2401.06066)
- [OLMoE: Open Mixture-of-Experts Language Models (Muennighoff et al., 2024)](https://arxiv.org/abs/2409.02060)

### 실무 적용
SaaS 제품에서 LLM 비용 구조를 설계할 때 "활성 파라미터 기준의 토큰당 비용"과 "전체 파라미터 기준의 메모리 비용"을 분리해 보는 관점을 줍니다. vLLM/SGLang의 Expert Parallelism, 배치 크기에 따른 MoE 처리량 변화 등은 셀프 호스팅 vs API 의사결정에서 실제 숫자로 이어집니다.

---

## 추천 읽기 순서
1. **GLU Variants (Paper 2)** — 4페이지, 30분이면 충분. 현대 LLM 블록의 FFN부터 정확히 이해합니다.
2. **Sparsely-Gated MoE (Paper 1)** — "그 FFN을 여러 개 두고 골라 쓰면?"이라는 질문의 원조 답. 게이팅과 로드 밸런싱 수식에 집중하세요.
3. **Mixtral of Experts (Paper 3)** — 앞의 두 아이디어가 실제 오픈 LLM으로 결합된 결과와 라우팅 분석을 확인합니다.

## 핵심 테이크어웨이
- **Q. 스케일링 법칙의 "파라미터 N"은 항상 비용과 같은가?** → 아니다. MoE는 전체 파라미터(지식 용량)와 활성 파라미터(연산 비용)를 분리한다.
- **Q. 현대 LLM 블록에서 가장 많이 바뀐 부분은?** → Attention 못지않게 FFN이다. ReLU FFN → SwiGLU FFN → SwiGLU 전문가의 MoE로 진화했다.
- **Q. MoE의 숨은 비용은?** → 로드 밸런싱, 학습 불안정성, 그리고 추론 시 메모리·통신 비용. 성능만큼 인프라 설계가 중요하다.
- **Q. 전문가는 정말 "도메인 전문가"인가?** → Mixtral 분석상 주로 구문·토큰 패턴에 특화된다. 직관적 이름에 속지 말 것.

## 다음 토픽과의 연결
다음 토픽은 **Instruction Tuning and RLHF**입니다. 오늘 본 아키텍처로 사전학습된 "베이스 모델"은 다음 토큰 예측기일 뿐이며, Mixtral-Instruct가 SFT + DPO를 거쳐 채팅 모델이 된 것처럼, 정렬(Alignment) 단계가 베이스 모델을 실제 제품에서 쓸 수 있는 어시스턴트로 바꾸는 과정을 다룹니다. 특히 MoE 모델에 RLHF/DPO를 적용할 때 라우팅 안정성이 어떤 이슈가 되는지 생각해 보며 넘어가면 좋습니다.
