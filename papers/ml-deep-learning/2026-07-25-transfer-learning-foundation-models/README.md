# Daily AI Paper Recommendations

> **Date:** 2026-07-25
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** Transfer Learning and Foundation Models

---

## Paper 1 (Classic): Universal Language Model Fine-tuning for Text Classification (ULMFiT)
- **Authors:** Jeremy Howard, Sebastian Ruder
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1801.06146
- **PDF:** [./ulmfit-universal-language-model-fine-tuning-howard-2018.pdf](./ulmfit-universal-language-model-fine-tuning-howard-2018.pdf)
- **Citation Count:** ~4,500 (approx.)

### 요약
ULMFiT는 컴퓨터 비전에서 당연시되던 "사전학습 후 파인튜닝" 패러다임을 NLP에 본격적으로 이식한 논문이다. 위키피디아 같은 대규모 일반 코퍼스로 언어모델을 학습한 뒤, 타깃 태스크 데이터로 언어모델을 미세조정하고, 마지막으로 분류기 레이어를 붙여 다운스트림 태스크를 푼다. 단 100개의 라벨만으로도 100배 많은 데이터로 처음부터 학습한 모델과 맞먹는 성능을 냈다.

### 핵심 기여
- 3단계 전이학습 레시피(일반 도메인 LM 사전학습 → 타깃 태스크 LM 파인튜닝 → 분류기 파인튜닝)를 제안해 NLP 전이학습을 표준화했다.
- Discriminative Fine-tuning: 레이어마다 다른 학습률을 적용해 하위 레이어의 일반 지식은 보존하고 상위 레이어만 크게 조정한다.
- Slanted Triangular Learning Rates(STLR)와 Gradual Unfreezing으로 파인튜닝 시 치명적 망각(catastrophic forgetting)을 완화했다.

### 이 논문이 중요한 이유
BERT/GPT 이전에 "언어모델 사전학습이 곧 범용 NLP 전이학습의 열쇠"임을 실증한 분수령 논문이다. 오늘날 모든 LLM 파인튜닝 워크플로우(사전학습된 모델을 받아 소량 데이터로 적응시키는 방식)의 사상적 원형이며, layer-wise learning rate·gradual unfreezing 같은 기법은 지금도 PEFT/파인튜닝 실무에서 재발견되고 있다. AI 엔지니어라면 "왜 처음부터 학습하지 않고 사전학습 모델을 미세조정하는가"에 대한 근거를 이 논문에서 얻는다.

### 사전 지식
- 언어모델(다음 토큰 예측)과 텍스트 분류의 차이
- LSTM/AWD-LSTM 기반 순환 신경망의 기본 구조
- 학습률 스케줄링, 파인튜닝, 과적합·망각 개념

### 관련 논문
- [Deep contextualized word representations (ELMo) (Peters et al., 2018)](https://arxiv.org/abs/1802.05365)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

### 실무 적용
도메인 특화 챗봇·문서 분류기를 만들 때 라벨이 부족한 상황에서 사전학습 LM을 파인튜닝하는 접근은 ULMFiT의 직계 후손이다. 예를 들어 사내 문서 분류, 감정분석, 스팸 필터를 소량 데이터로 구축할 때 "일반 LM → 도메인 적응 → 태스크 헤드"라는 3단계 파이프라인을 그대로 쓴다. Hugging Face의 `Trainer` 기반 파인튜닝, 점진적 언프리징 전략이 대표적 계승 사례다.

---

## Paper 2 (Classic): Big Transfer (BiT): General Visual Representation Learning
- **Authors:** Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Joan Puigcerver, Jessica Yung, Sylvain Gelly, Neil Houlsby
- **Year:** 2019 (ECCV 2020)
- **arXiv:** https://arxiv.org/abs/1912.11370
- **PDF:** [./big-transfer-bit-kolesnikov-2019.pdf](./big-transfer-bit-kolesnikov-2019.pdf)
- **Citation Count:** ~3,200 (approx.)

### 요약
BiT는 비전 분야에서 "대규모 지도학습 사전학습 + 단순한 전이 레시피"만으로 얼마나 강력한 범용 표현을 얻을 수 있는지를 체계적으로 규명한 논문이다. JFT-300M 같은 초대형 데이터로 ResNet을 키워 사전학습하고, 타깃 태스크에는 복잡한 정규화 없이 간단한 휴리스틱(BiT-HyperRule)으로 파인튜닝한다. 클래스당 1개 예시부터 100만 예시까지 폭넓은 데이터 규모에서 20개 이상 데이터셋에 걸쳐 강력한 성능을 보였다.

### 핵심 기여
- 대규모 사전학습 데이터·모델 규모·학습 시간을 키우면 전이 성능이 일관되게 향상됨을 대규모 실험으로 입증했다(스케일링의 힘).
- Group Normalization + Weight Standardization 조합이 대규모 배치 사전학습과 전이에서 Batch Norm보다 안정적임을 보였다.
- BiT-HyperRule: 데이터셋 크기에 따라 학습 스텝·해상도·MixUp만 규칙적으로 정하는 단순한 파인튜닝 레시피로 태스크별 하이퍼파라미터 튜닝 부담을 없앴다.

### 이 논문이 중요한 이유
"파운데이션 모델"이라는 용어가 등장하기 전에, 하나의 사전학습 백본을 수십 개 태스크에 재사용한다는 파운데이션 모델의 핵심 논리를 비전에서 선명하게 보여준 논문이다. 특히 few-shot 전이(클래스당 소수 예시)에서의 강력함은 오늘날 데이터가 부족한 실무 환경에서 사전학습 백본이 왜 필수인지를 설명한다. 스케일과 정규화 선택이 전이 품질을 좌우한다는 통찰은 이후 ViT·CLIP·멀티모달 파운데이션 모델 설계로 이어진다.

### 사전 지식
- CNN/ResNet 구조와 ImageNet 사전학습 개념
- Batch Norm vs Group Norm, Weight Standardization의 차이
- 파인튜닝, few-shot 전이, 데이터 규모(scale)와 일반화의 관계

### 관련 논문
- [Exploring the Limits of Weakly Supervised Pretraining (Mahajan et al., 2018)](https://arxiv.org/abs/1805.00932)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition (ViT) (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)

### 실무 적용
이미지 분류·검출·의료영상처럼 라벨이 비싼 도메인에서 사전학습된 백본(BiT, 이후의 ViT/DINOv2 등)을 받아 소량 데이터로 파인튜닝하는 전략은 산업 현장의 기본기다. BiT가 정립한 "큰 백본 + 단순 전이 레시피" 원칙은 오늘날 timm·Hugging Face의 사전학습 비전 백본을 그대로 가져다 쓰는 워크플로우, 그리고 데이터 규모별로 파인튜닝 설정을 자동화하는 관행으로 살아 있다.

---

## Paper 3 (Recent): Qwen2.5-VL Technical Report
- **Authors:** Qwen Team, Alibaba Group
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.13923
- **PDF:** [./qwen2.5-vl-technical-report-qwen-2025.pdf](./qwen2.5-vl-technical-report-qwen-2025.pdf)
- **Citation Count:** ~700+ (approx., 급증 중)

### 요약
Qwen2.5-VL은 이미지·비디오·문서·에이전트 조작까지 아우르는 오픈 비전-언어 파운데이션 모델 시리즈다. 동적 해상도(native dynamic-resolution) ViT를 처음부터 학습하고 Window Attention을 도입해 원본 해상도를 유지하면서 연산량을 크게 줄였다. 절대 시간 인코딩(absolute time encoding)으로 수 시간 길이의 영상에서도 초 단위 이벤트 위치 지정이 가능하며, GUI 조작·문서 파싱·객체 그라운딩 등 에이전트형 태스크에서 강력한 성능을 보인다.
