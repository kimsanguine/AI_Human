# Daily AI Paper Recommendations

> **Date:** 2026-11-12
> **Module:** Module 4: NLP and Speech Data
> **Topic:** Attention Mechanism and Transformer

---

## Paper 1 (Classic): Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
- **Authors:** Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio
- **Year:** 2015 (ICML 2015)
- **arXiv:** https://arxiv.org/abs/1502.03044
- **PDF:** [./show-attend-tell-xu-2015.pdf](./show-attend-tell-xu-2015.pdf)
- **Citation Count:** ~13,000+

### 요약
CNN이 뽑은 이미지 영역 특징 위에 LSTM 디코더를 올리고, 단어를 하나 생성할 때마다 "이미지의 어느 영역을 볼지"를 어텐션으로 고르는 이미지 캡셔닝 모델이다. 결정적(soft) 어텐션과 확률적(hard) 어텐션 두 방식을 제안하고, 어텐션 맵을 시각화해 모델이 실제로 무엇을 보고 단어를 만드는지 보여준다.

### 핵심 기여
- Bahdanau식 어텐션을 번역(텍스트→텍스트)에서 비전-언어(이미지→텍스트)로 확장해 어텐션이 모달리티에 무관한 범용 메커니즘임을 입증
- Soft attention(미분 가능한 가중 평균)과 Hard attention(REINFORCE 기반 샘플링)을 정식화하고 비교 — 오늘날 "soft/hard attention" 용어의 출처
- 어텐션 가중치 시각화를 통해 "모델이 어디를 보고 있는가"라는 해석 가능성(interpretability) 도구를 대중화
- Flickr8k/30k, MS COCO에서 당시 최고 수준의 BLEU/METEOR 달성

### 이 논문이 중요한 이유
Transformer 이전에 "어텐션 = 동적으로 정보를 골라 읽는 소프트 검색"이라는 직관을 가장 선명하게 보여준 논문이다. 오늘날의 Vision-Language 모델(LLaVA, Flamingo 등)이 이미지 패치 토큰에 텍스트 토큰이 cross-attention하는 구조의 직계 조상이며, AI 아바타·더빙처럼 멀티모달 제품을 다루는 엔지니어에게 "어텐션은 텍스트 전용이 아니다"라는 관점을 처음 심어준다.

### 사전 지식
- CNN 특징 맵(feature map)과 LSTM 디코더의 기본 구조
- Bahdanau attention(정렬 점수 → softmax → 가중합)
- REINFORCE(정책 경사)의 기본 개념 (hard attention 이해용)

### 관련 논문
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014)](https://arxiv.org/abs/1409.0473)
- [Show and Tell: A Neural Image Caption Generator (Vinyals et al., 2014)](https://arxiv.org/abs/1411.4555)
- [Recurrent Models of Visual Attention (Mnih et al., 2014)](https://arxiv.org/abs/1406.6247)
- [Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198)

### 실무 적용
이미지·영상 캡셔닝, 접근성용 대체 텍스트 자동 생성, 영상 기반 AI 더빙에서 화면 맥락을 반영한 번역 등에 쓰이는 cross-attention 구조의 출발점이다. 또한 어텐션 맵 시각화는 지금도 멀티모달 모델의 오류 분석(모델이 엉뚱한 영역을 보고 있지 않은지) 도구로 활용된다.

---

## Paper 2 (Classic): Big Bird: Transformers for Longer Sequences
- **Authors:** Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed
- **Year:** 2020 (NeurIPS 2020)
- **arXiv:** https://arxiv.org/abs/2007.14062
- **PDF:** [./big-bird-zaheer-2020.pdf](./big-bird-zaheer-2020.pdf)
- **Citation Count:** ~2,500+

### 요약
전체 토큰 쌍을 보는 O(n²) 어텐션 대신, 슬라이딩 윈도우(local) + 랜덤(random) + 글로벌(global) 토큰 세 가지 희소 패턴을 결합한 O(n) 어텐션을 제안한다. 이 희소 어텐션이 원래 Transformer의 이론적 성질(범용 근사자, 튜링 완전성)을 유지함을 증명하고, 같은 하드웨어에서 최대 8배 긴 시퀀스를 처리해 QA·요약·유전체 서열 과제에서 성능을 끌어올린다.

### 핵심 기여
- Local + Random + Global의 3요소 희소 어텐션 패턴 설계 (그래프 이론의 확장 그래프·작은 세상 네트워크 관점)
- 희소 어텐션도 full attention의 표현력(universal approximator, Turing complete)을 잃지 않는다는 이론적 증명 — 특히 글로벌 토큰의 필요성을 이론으로 정당화
- 4,096 토큰 문맥으로 Natural Questions, TriviaQA, arXiv/PubMed 장문 요약 등에서 SOTA, DNA 서열 등 비언어 도메인까지 적용

### 이 논문이 중요한 이유
이전 사이클에서 본 Sparse Transformers, Longformer가 "경험적으로 잘 되는 희소 패턴"이었다면, Big Bird는 "왜 희소해도 괜찮은가"에 대한 이론적 답을 준 논문이다. 2025년의 Native Sparse Attention, MoBA 같은 최신 희소 어텐션도 결국 "모든 토큰이 볼 필요는 없지만 정보가 몇 홉 안에 전달될 경로는 보장해야 한다"는 Big Bird의 그래프 관점 위에 있다.

### 사전 지식
- Scaled Dot-Product Attention과 O(n²) 복잡도의 원인
- Longformer의 sliding window + global attention 개념
- 그래프 기본 개념(노드, 엣지, 경로 길이)

### 관련 논문
- [Longformer: The Long-Document Transformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [Generating Long Sequences with Sparse Transformers (Child et al., 2019)](https://arxiv.org/abs/1904.10509)
- [ETC: Encoding Long and Structured Inputs in Transformers (Ainslie et al., 2020)](https://arxiv.org/abs/2004.08483)
- [Native Sparse Attention (Yuan et al., 2025)](https://arxiv.org/abs/2502.11089)

### 실무 적용
계약서·논문·회의록처럼 긴 문서를 한 번에 인코딩해야 하는 분류/QA/요약 서비스에서 유용하다. Hugging Face에 BigBird-RoBERTa, BigBird-Pegasus 체크포인트가 공개되어 있어, LLM 호출 비용이 부담되는 장문 분류·추출 파이프라인에 경량 인코더로 투입할 수 있다. 어떤 희소 패턴을 고를지 판단할 때 "글로벌 토큰(예: [CLS], 질문 토큰)을 무엇으로 둘 것인가"가 실무적 핵심 설계 포인트다.

---

## Paper 3 (Recent): Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free
- **Authors:** Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin
- **Year:** 2025 (NeurIPS 2025)
- **arXiv:** https://arxiv.org/abs/2505.06708
- **PDF:** [./gated-attention-qiu-2025.pdf](./gated-attention-qiu-2025.pdf)
- **Citation Count:** ~100+ (2025년 발표, 빠르게 증가 중)

### 요약
Scaled Dot-Product Attention(SDPA) 출력 뒤에 헤드별 sigmoid 게이트 하나를 곱하는 아주 단순한 변경이 성능·학습 안정성·스케일링을 일관되게 개선함을 보인다. 15B MoE와 1.7B dense 모델 30여 개 변형을 3.5T 토큰으로 학습해 체계적으로 비교했으며, 이 게이트는 Qwen3-Next 모델에 실제 채택되었다.

### 핵심 기여
- 게이트 위치(Q/K/V 뒤, SDPA 뒤, 출력 projection 뒤 등)와 형태를 대규모로 비교해 "SDPA 출력 직후 + 헤드별 + sigmoid + 곱셈"이 최적임을 실증
- 효과의 원인을 두 가지로 분석: ① V–O projection의 저랭크 선형 사상에 비선형성 추가 ② 쿼리 의존적 희소 게이팅으로 불필요한 출력 억제
- Attention sink 제거: 첫 토큰에 몰리던 어텐션 비중을 평균 약 46.7% → 4.8%로 줄이고, 대규모 활성값(massive activation)도 완화
- 학습 손실 스파이크 감소로 더 큰 학습률 사용 가능, 롱컨텍스트 외삽(RULER)에서 10점 이상 향상

### 이 논문이 중요한 이유
2017년 이후 거의 바뀌지 않은 "어텐션 블록 자체"에 대해, 작은 구조 변경이 대형 LLM에서 어떤 효과를 내는지 대규모 ablation으로 보여준 드문 연구이며 NeurIPS 2025 최우수 논문 중 하나로 선정되었다. 특히 StreamingLLM 등에서 "어쩔 수 없는 현상"으로 다뤄지던 attention sink가 게이트 부재로 생긴 부산물일 수 있다는 해석을 제시해, 양자화·KV 캐시 압축·롱컨텍스트 문제를 바라보는 관점을 바꾼다.

### 사전 지식
- Multi-head attention의 Q/K/V/O projection 구조
- Attention sink 현상과 StreamingLLM
- GLU/SwiGLU 등 게이팅 메커니즘
- MoE LLM의 기본 개념

### 관련 논문
- [Efficient Streaming Language Models with Attention Sinks (Xiao et al., 2023)](https://arxiv.org/abs/2309.17453)
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [Differential Transformer (Ye et al., 2024)](https://arxiv.org/abs/2410.05258)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule (Yang et al., 2024)](https://arxiv.org/abs/2412.06464)

### 실무 적용
직접 사전학습하지 않더라도, attention sink가 적은 모델은 활성값 이상치가 줄어 INT8/FP8 양자화와 KV 캐시 압축이 쉬워지고, 긴 대화·긴 문서에서의 성능 저하도 줄어든다. 오픈 모델(Qwen3-Next 계열 등) 선택 시 "게이트 어텐션 채택 여부"를 롱컨텍스트 안정성과 서빙 효율의 판단 지표로 삼을 수 있으며, 자체 소형 모델을 학습하는 팀이라면 코드 몇 줄로 적용 가능한 저비용·고효과 개선안이다.

---

## 추천 읽기 순서
1. **Show, Attend and Tell** — "어텐션 = 무엇을 볼지 고르는 소프트 검색"이라는 직관을 시각적으로 먼저 잡는다 (어텐션 맵 그림 위주로 읽기).
2. **Big Bird** — 그 "고르기"를 전부 다 할 필요는 없다는 것, 즉 희소 어텐션이 왜 이론적으로도 충분한지 이해한다 (섹션 2 아키텍처, 섹션 3 이론은 결론 위주).
3. **Gated Attention** — 고른 결과를 "얼마나 내보낼지"를 게이트로 조절하면 무엇이 달라지는지 본다 (게이트 위치 비교표와 attention sink 분석 섹션 중심).

## 핵심 테이크어웨이
- 어텐션은 세 단계로 분해해 볼 수 있다: **어디를 볼지(점수) → 얼마나 많이 볼지(희소성) → 본 것을 얼마나 내보낼지(게이팅)**. 오늘 세 논문이 각각 이 세 단계를 대표한다.
- 어텐션은 모달리티 중립적 메커니즘이다. 이미지 영역이든, 텍스트 토큰이든, 오디오 프레임이든 "검색 가능한 메모리"로 만들면 같은 구조가 적용된다.
- 희소성은 근사가 아니라 설계다. Big Bird의 글로벌 토큰처럼 "정보가 흐를 경로"만 보장하면 표현력을 잃지 않는다.
- softmax는 합이 1이어야 하므로 "아무것도 보지 않기"를 표현하지 못한다. Attention sink는 그 부작용이고, 게이트는 출력 측에서 "0을 낼 자유"를 돌려준다.
- 탐구 질문: 우리 서비스에서 긴 입력의 품질 저하가 일어날 때, 그 원인은 "못 보는 것(희소/길이 한계)"인가, "엉뚱한 곳에 몰리는 것(sink)"인가? 두 원인에 대한 처방은 다르다.

## 다음 토픽과의 연결
다음 토픽 **BERT and Pre-trained Language Models**에서는 오늘 본 어텐션 블록을 쌓아 "무엇을 예측하도록 사전학습하는가"가 중심이 된다. Big Bird가 실제로 BERT/RoBERTa 체크포인트에서 출발해 긴 문서용 인코더로 확장되었다는 점, 그리고 ModernBERT 같은 최신 인코더가 로컬/글로벌 교대 어텐션을 채택한 점을 연결해서 보면, "어텐션 설계 선택"이 사전학습 모델의 용도(문서 길이, 비용)를 결정한다는 흐름을 확인할 수 있다.
