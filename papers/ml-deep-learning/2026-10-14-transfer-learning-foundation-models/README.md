# Daily AI Paper Recommendations

> **Date:** 2026-10-14
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): Parameter-Efficient Transfer Learning for NLP
- **Authors:** Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly
- **Year:** 2019 (ICML 2019)
- **arXiv:** https://arxiv.org/abs/1902.00751
- **PDF:** [./adapters-houlsby-2019.pdf](./adapters-houlsby-2019.pdf)
- **Citation Count:** ~5,000+

### 요약
사전학습된 BERT를 태스크마다 통째로 파인튜닝하면 태스크 수만큼 거대한 모델 사본이 생긴다. 이 논문은 Transformer 각 레이어에 작은 병목(bottleneck) 구조의 **Adapter 모듈**만 끼워 넣고, 원래 가중치는 얼린 채 Adapter만 학습하는 방식을 제안했다. 태스크당 파라미터를 약 3.6%만 추가하고도 GLUE에서 전체 파인튜닝 대비 0.4% 이내의 성능을 달성했다.

### 핵심 기여
- Down-projection → 비선형 → Up-projection + residual로 이루어진 Adapter 모듈 설계 (초기에는 거의 identity로 동작하도록 near-zero 초기화)
- "하나의 공유 백본 + 태스크별 작은 모듈"이라는 **파라미터 효율적 전이학습(PEFT)** 패러다임을 정립
- 26개 텍스트 분류 태스크에서 파라미터 수 대비 성능 트레이드오프를 체계적으로 비교, 상위 레이어 파인튜닝보다 훨씬 효율적임을 실증
- 어떤 레이어의 Adapter가 중요한지(상위 레이어일수록 중요) ablation으로 분석

### 이 논문이 중요한 이유
LoRA, Prefix-Tuning, Prompt Tuning, QLoRA로 이어지는 PEFT 계보의 출발점이다. "파운데이션 모델은 얼려두고, 작은 부분만 바꿔서 특화시킨다"는 사고방식은 오늘날 LLM 커스터마이징의 기본값이 되었다. LoRA를 쓰는 AI 엔지니어라면 왜 low-rank/bottleneck 구조가 작동하는지, 그리고 Adapter가 추론 지연(latency)을 늘린다는 한계가 LoRA를 낳았다는 맥락을 알아야 한다.

### 사전 지식
- Transformer 블록 구조(Multi-head attention, FFN, LayerNorm, residual)
- BERT 사전학습 및 표준 파인튜닝 방식
- 파라미터 수, 메모리, 저장 비용 관점의 모델 배포 개념

### 관련 논문
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021)](https://arxiv.org/abs/2101.00190)
- [AdapterFusion: Non-Destructive Task Composition for Transfer Learning (Pfeiffer et al., 2020)](https://arxiv.org/abs/2005.00247)
- [Towards a Unified View of Parameter-Efficient Transfer Learning (He et al., 2021)](https://arxiv.org/abs/2110.04366)

### 실무 적용
B2B SaaS에서 고객사별/도메인별 모델을 운영할 때, 고객마다 풀 모델을 복제하는 대신 공유 백본 하나에 고객별 Adapter(또는 LoRA)를 붙이면 저장·서빙 비용이 급감한다. vLLM, S-LoRA 같은 서빙 엔진의 multi-adapter 기능이 바로 이 아이디어의 확장이다. 또한 음성(TTS/ASR)·비전 모델에서도 화자/언어별 Adapter를 두는 방식으로 적은 데이터로 빠르게 특화 모델을 만들 수 있다.

---

## Paper 2 (Classic): Do Better ImageNet Models Transfer Better?
- **Authors:** Simon Kornblith, Jonathon Shlens, Quoc V. Le
- **Year:** 2018 (arXiv) / 2019 (CVPR 2019)
- **arXiv:** https://arxiv.org/abs/1805.08974
- **PDF:** [./better-imagenet-transfer-kornblith-2019.pdf](./better-imagenet-transfer-kornblith-2019.pdf)
- **Citation Count:** ~1,800+

### 요약
"ImageNet에서 더 잘하는 모델이 다른 태스크로 전이해도 더 잘하는가?"라는 컴퓨터비전 전이학습의 암묵적 가정을 16개 아키텍처 × 12개 데이터셋으로 정면 검증한 논문이다. 고정 특징(fixed feature)과 파인튜닝 모두에서 ImageNet 정확도와 전이 성능은 강한 상관(r≈0.96~0.99)을 보였다. 하지만 정규화 기법(label smoothing, dropout 등)이 ImageNet 정확도는 올려도 고정 특징의 전이 품질은 오히려 떨어뜨리고, 세분화된(fine-grained) 소규모 데이터셋에서는 사전학습의 이점이 기대보다 작다는 반전도 함께 보고했다.

### 핵심 기여
- 대규모·통제된 실험으로 "업스트림 성능 → 다운스트림 성능" 상관관계를 정량화
- 학습 시 정규화 설정이 penultimate 특징의 전이 가능성을 해칠 수 있음을 발견 (벤치마크 점수 ≠ 표현 품질)
- 일부 데이터셋(Stanford Cars, FGVC Aircraft)에서는 처음부터 학습한 모델과 파인튜닝 모델의 격차가 작음을 보여 사전학습의 한계를 명시
- 전이학습 평가 프로토콜(linear probe vs. fine-tuning)의 기준을 제시

### 이 논문이 중요한 이유
"어떤 파운데이션 모델을 고를 것인가?"는 AI 엔지니어가 매일 내리는 결정이다. 이 논문은 리더보드 점수가 내 다운스트림 태스크 성능을 대체로 예측하지만 **항상은 아니다**라는 점, 그리고 그 예외가 어디서 생기는지를 데이터로 보여준다. 이후 BiT, CLIP, DINO 계열의 linear probe 평가 관행도 이 논문의 문제의식 위에 있다.

### 사전 지식
- CNN 아키텍처 계보(Inception, ResNet, NASNet 등)
- 고정 특징 추출(linear probe)과 파인튜닝의 차이
- 상관계수와 통계적 비교의 기본

### 관련 논문
- [How transferable are features in deep neural networks? (Yosinski et al., 2014)](https://arxiv.org/abs/1411.1792)
- [Rethinking ImageNet Pre-training (He et al., 2018)](https://arxiv.org/abs/1811.08883)
- [Big Transfer (BiT): General Visual Representation Learning (Kolesnikov et al., 2019)](https://arxiv.org/abs/1912.11370)
- [When Does Label Smoothing Help? (Müller, Kornblith & Hinton, 2019)](https://arxiv.org/abs/1906.02629)

### 실무 적용
백본을 고를 때 공개 벤치마크 1위 모델을 바로 채택하기보다, 내 데이터로 **linear probe 기반의 빠른 후보 비교**를 먼저 돌리는 것이 안전하다. 특히 도메인이 특수한 경우(의료 영상, 아바타 얼굴, 제품 사진 등)에는 사전학습 이점이 줄어들 수 있으므로 데이터 규모별 학습 곡선을 그려 투자 대비 효과를 판단해야 한다. LLM 선택에서도 "MMLU 점수 높은 모델이 우리 태스크에서도 낫다"는 가정을 자체 평가셋으로 검증하는 것과 같은 원리다.

---

## Paper 3 (Recent): DINOv3
- **Authors:** Oriane Siméoni, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, et al. (Meta AI), Piotr Bojanowski
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2508.10104
- **PDF:** [./dinov3-simeoni-2025.pdf](./dinov3-simeoni-2025.pdf)
- **Citation Count:** 신규 논문 (2025년 8월 공개, 빠르게 인용 증가 중)

### 요약
레이블 없이 자기지도학습(SSL)만으로 범용 비전 파운데이션 모델을 만드는 DINO 계열의 3세대 기술 보고서다. 데이터와 모델(최대 7B 파라미터 ViT)을 크게 키우면서 생기는 고질적 문제 — 긴 학습에서 dense feature(패치 단위 특징) 품질이 무너지는 현상 — 를 **Gram anchoring**이라는 새로운 정규화로 해결했다. 그 결과 하나의 얼린(frozen) 백본으로 분류, 분할, 깊이 추정, 3D 대응, 위성 영상 등 다양한 태스크에서 특화 모델에 필적하거나 능가하는 성능을 보였다.

### 핵심 기여
- 7B 규모 ViT와 대규모 큐레이션 데이터로 SSL 스케일링을 밀어붙임
- Gram anchoring: 패치 특징 간 유사도 구조(Gram matrix)를 초기 teacher에 고정해 dense feature 열화를 방지
- 고해상도 적응 및 다양한 크기의 모델로의 증류(distillation) 패밀리 제공 → 실무 배포 가능
- 위성/항공 영상 등 웹 이미지가 아닌 도메인까지 동일 레시피 적용

### 이 논문이 중요한 이유
LLM 시대의 "파운데이션 모델"이 텍스트에만 해당하는 개념이 아님을 보여준다. 텍스트 지도(CLIP류) 없이도 픽셀만으로 범용 표현을 학습할 수 있고, **백본을 얼린 채 가벼운 head만 붙이는** 전이 방식이 대규모에서 실제로 작동한다는 것을 증명했다. Paper 1(얼린 백본 + 작은 모듈)과 Paper 2(업스트림 품질이 다운스트림을 결정하는가)의 질문에 대한 2025년식 답이기도 하다.

### 사전 지식
- Vision Transformer(ViT)와 패치 토큰 개념
- 자기지도학습의 teacher-student / self-distillation 구조 (DINO, iBOT, DINOv2)
- dense prediction 태스크(semantic segmentation, depth estimation)

### 관련 논문
- [DINOv2: Learning Robust Visual Features without Supervision (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- [Emerging Properties in Self-Supervised Vision Transformers / DINO (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)
- [Vision Transformers Need Registers (Darcet et al., 2023)](https://arxiv.org/abs/2309.16588)
- [SigLIP 2: Multilingual Vision-Language Encoders (Tschannen et al., 2025)](https://arxiv.org/abs/2502.14786)

### 실무 적용
레이블 데이터가 부족한 비전 기능(예: 아바타 얼굴 영역 분할, 영상 장면 검색, 제품 이미지 유사도 검색)을 만들 때 DINOv3 백본을 얼리고 linear/light head만 학습하면 수천 장 수준의 데이터로도 강력한 베이스라인을 빠르게 확보할 수 있다. 패치 단위 특징이 좋기 때문에 비디오 객체 추적, 영상 편집용 마스크 생성 등 dense 태스크에서 특히 유용하며, 증류된 소형 모델로 엣지/실시간 서빙까지 고려할 수 있다.

---

## 추천 읽기 순서
1. **Do Better ImageNet Models Transfer Better? (Kornblith 2019)** — "전이학습은 정말 작동하는가, 언제 작동하지 않는가"라는 질문으로 문제의식을 먼저 세운다.
2. **Parameter-Efficient Transfer Learning for NLP (Houlsby 2019)** — 백본을 얼리고 작은 모듈만 학습하는 "효율적 전이"의 방법론을 익힌다.
3. **DINOv3 (Siméoni 2025)** — 두 질문이 스케일업된 2025년의 비전 파운데이션 모델에서 어떻게 답해지는지 확인한다.

## 핵심 테이크어웨이
- **업스트림 품질은 대체로 다운스트림을 예측하지만, 예외(정규화 설정·특수 도메인)를 항상 자체 검증하라.**
- **파운데이션 모델 활용의 기본값은 "얼린 백본 + 작은 학습 모듈"이다.** Adapter → LoRA → frozen-backbone head로 이어지는 흐름은 비용·배포 측면에서 압도적으로 유리하다.
- **벤치마크 점수와 표현 품질은 다르다.** Kornblith가 발견한 정규화의 역효과와 DINOv3의 dense feature 열화 문제는 같은 교훈을 준다: 무엇을 측정하느냐가 무엇을 얻느냐를 결정한다.
- PM 관점의 질문: "우리 기능에 필요한 건 모델 전체의 재학습인가, 아니면 공유 백본 위의 작은 특화 레이어인가?"

## 다음 토픽과의 연결
다음은 **Module 4: NLP and Speech Data — Word Embeddings and Representation Learning**이다. 오늘 다룬 "좋은 표현(representation)을 미리 학습해 두고 다른 태스크로 옮긴다"는 전이학습의 핵심 아이디어는, 텍스트 영역에서 Word2Vec·GloVe 같은 단어 임베딩에서 처음 대중화되었다. 비전의 사전학습 특징과 NLP의 임베딩이 같은 원리라는 연결고리를 염두에 두고 읽으면 좋다.
