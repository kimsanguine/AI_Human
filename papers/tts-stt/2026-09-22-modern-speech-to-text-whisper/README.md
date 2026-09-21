# Daily AI Paper Recommendations

> **Date:** 2026-09-22
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Modern Speech-to-Text Whisper and Beyond

---

## Paper 1 (Classic): Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages
- **Authors:** Yu Zhang, Wei Han, James Qin, Yongqiang Wang, Ankur Bapna, Zhehuai Chen, Nanxin Chen, Bo Li, Vera Axelrod, Gary Wang, Zhong Meng, Ke Hu, Andrew Rosenberg, Rohit Prabhavalkar, Daniel S. Park, Parisa Haghani, Jason Riesa, Ginger Perng, Hagen Soltau, Trevor Strohman, Bhuvana Ramabhadran, Tara Sainath, Pedro Moreno, Chung-Cheng Chiu, Johan Schalkwyk, Françoise Beaufays, Yonghui Wu
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2303.01037
- **PDF:** [./google-usm-zhang-2023.pdf](./google-usm-zhang-2023.pdf)
- **Citation Count:** ~700+

### 요약
Google USM(Universal Speech Model)은 2B 파라미터 Conformer 인코더를 1,200만 시간 분량의 라벨 없는 다국어 오디오와 280억 문장의 텍스트로 사전학습한 뒤, 소량의 라벨 데이터로만 미세조정해 100개 이상 언어의 음성 인식을 수행하는 모델이다. 핵심은 BEST-RQ 기반 자기지도 사전학습 → 텍스트 주입(text injection) 중간 학습 → 최소한의 지도 미세조정으로 이어지는 3단계 파이프라인이다. 결과적으로 Whisper보다 훨씬 적은 라벨 데이터로 YouTube 18개 언어 및 다수의 다국어 벤치마크에서 더 낮은 WER를 달성했다.

### 핵심 기여
- BEST-RQ(랜덤 프로젝션 양자화기)를 이용한 대규모 자기지도 사전학습이 wav2vec 2.0류 대비 훨씬 단순하면서도 스케일링에 강하다는 것을 실증
- 텍스트-온리 데이터를 인코더에 주입하는 중간 학습 단계(text injection)로 라벨 음성이 거의 없는 저자원 언어의 성능을 끌어올림
- "사전학습은 크게, 지도학습은 작게" 라는 ASR 스케일링 레시피를 정립 — Whisper의 대규모 약지도(weak supervision) 노선과 대비되는 또 하나의 축

### 이 논문이 중요한 이유
Whisper만 알고 있으면 "ASR = 대량의 (오디오, 텍스트) 쌍을 긁어모아 지도학습" 이라는 단일 관점에 갇히기 쉽다. USM은 같은 목표를 라벨 없는 오디오 + 텍스트 코퍼스라는 전혀 다른 자원으로 달성한다. AI 엔지니어가 실제로 새로운 언어·도메인(콜센터, 의료, 사내 용어)을 지원해야 할 때, 라벨링 예산을 어디에 쓸지 결정하는 사고 틀을 제공한다. 오늘 읽을 Paper 3(Omnilingual ASR)의 직접적인 지적 조상이기도 하다.

### 사전 지식
- Conformer 인코더 구조(Self-attention + Convolution 결합)
- 자기지도 학습의 마스킹 예측 패러다임 (BERT의 MLM, wav2vec 2.0의 contrastive)
- RNN-T / CTC 디코딩의 차이와 WER 평가 방식
- 저자원(low-resource) 언어 문제와 데이터 불균형의 개념

### 관련 논문
- [BEST-RQ: Self-supervised Learning with Random-projection Quantizer for Speech Recognition (Chiu et al., 2022)](https://arxiv.org/abs/2202.01855)
- [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)
- [Scaling Speech Technology to 1,000+ Languages / MMS (Pratap et al., 2023)](https://arxiv.org/abs/2305.13516)

### 실무 적용
Google의 YouTube 자동 자막과 Cloud Speech-to-Text 다국어 지원의 기반 기술이다. 실무에서는 "USM 스타일 레시피"를 축소 적용하는 형태로 쓰인다 — 즉, 자사 도메인의 라벨 없는 통화·회의 녹음 수천 시간으로 기존 인코더를 계속 사전학습(continued pre-training)한 뒤, 사람이 검수한 수십 시간짜리 소량 데이터로만 미세조정하는 방식이다. 라벨링 비용을 10분의 1 수준으로 줄이면서 도메인 WER를 유의미하게 낮출 수 있다.

---

## Paper 2 (Classic): data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language
- **Authors:** Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2202.03555
- **PDF:** [./data2vec-baevski-2022.pdf](./data2vec-baevski-2022.pdf)
- **Citation Count:** ~1,500+

### 요약
data2vec은 음성·비전·텍스트라는 세 가지 모달리티에 대해 **동일한 학습 목표**를 사용하는 자기지도 학습 프레임워크다. 모달리티별로 다른 토큰·이산 단위·contrastive 목표를 설계하는 대신, 교사(teacher) 모델의 상위 레이어 표현을 평균낸 **잠재 표현(latent representation)을 마스킹된 위치에서 회귀(regression)** 하도록 학생(student) 모델을 학습시킨다. 교사는 학생의 EMA(지수이동평균)로 갱신된다. 음성에서는 wav2vec 2.0/HuBERT를, 비전에서는 BEiT를, 텍스트에서는 RoBERTa를 각각 상회하거나 필적하는 성능을 보였다.

### 핵심 기여
- 모달리티 독립적(modality-agnostic) 자기지도 목표 제시 — 이산 단위(codebook, k-means 클러스터) 설계 없이 연속 잠재 표현을 직접 예측
- 자기증류(self-distillation) + EMA 교사 구조를 음성 표현 학습에 성공적으로 적용, 표현 붕괴(collapse)를 막는 정규화 기법(instance norm, 상위 K개 레이어 평균) 제시
- 단일 레시피로 세 모달리티 SOTA에 근접함으로써, 이후 멀티모달 음성-언어 모델의 통합 표현 학습 흐름을 열었음

### 이 논문이 중요한 이유
현재의 음성 LLM(Qwen-Audio, Voxtral, GPT-4o 계열 음성 입력)은 결국 "오디오를 언어 모델이 이해할 수 있는 연속 표현으로 어떻게 바꿀 것인가"의 문제다. data2vec은 그 답을 모달리티에 특화된 트릭이 아니라 일반 원리로 제시한 첫 논문에 가깝다. AI 엔지니어 입장에서는 wav2vec 2.0(contrastive) → HuBERT(이산 클러스터) → data2vec(연속 잠재 회귀)로 이어지는 계보를 이해해야 자사 도메인에 어떤 사전학습을 붙일지 판단할 수 있다.

### 사전 지식
- wav2vec 2.0의 contrastive 목표와 HuBERT의 k-means 타깃 방식
- 자기증류(self-distillation)와 EMA 교사-학생 구조 (BYOL, DINO)
- 마스킹 기반 표현 학습(MLM)과 표현 붕괴(representation collapse) 문제
- Transformer 레이어별 표현이 서로 다른 정보를 담는다는 사실(probing 관점)

### 관련 논문
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)
- [data2vec 2.0: Efficient Self-supervised Learning with Contextualized Target Representations (Baevski et al., 2022)](https://arxiv.org/abs/2212.07525)

### 실무 적용
data2vec 계열 인코더는 라벨이 거의 없는 도메인에서 음성 임베딩 백본으로 널리 쓰인다. 예: 화자 분리(diarization), 감정·의도 분류, 음성 검색(speech retrieval), 키워드 스포팅 등 ASR이 아닌 다운스트림 태스크. 또한 "하나의 사전학습 레시피로 오디오·이미지·텍스트 인코더를 모두 관리한다"는 접근은 멀티모달 제품을 운영하는 팀의 MLOps 복잡도를 크게 줄여준다.

---

## Paper 3 (Recent): Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages
- **Authors:** Omnilingual ASR team (Meta AI) — Gil Keren, Artyom Kozhevnikov, Yen Meng, Christophe Ropers, Matthew Setzler, Skyler Wang, Ife Adebara, Michael Auli, Can Balioglu, Vineel Pratap, Yu-An Chung, Jean Maillard, Alexandre Mourachko, Mary Williamson, et al.
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2511.09690
- **PDF:** [./omnilingual-asr-meta-2025.pdf](./omnilingual-asr-meta-2025.pdf)
- **Citation Count:** 급속 증가 중 (2026년 9월 기준 100+ 추정)

### 요약
Omnilingual ASR은 1,600개 이상의 언어를 지원하는 최초의 대규모 오픈소스 ASR 시스템으로, 그중 500개 이상은 지금까지 어떤 ASR 시스템도 지원하지 않던 언어다. 7B 파라미터까지 자기지도 사전학습을 확장한 음성 인코더 위에, LLM에서 영감을 받은 인코더-디코더 구조를 올려 **제로샷/퓨샷 일반화**를 가능하게 했다. 가장 큰 설계 전환은 "새 언어를 추가하려면 모델을 재학습해야 한다"는 전제를 깨고, 커뮤니티가 **소수의 오디오-텍스트 샘플만으로 미지원 언어를 추가**할 수 있는 확장성(extensibility)을 1급 요구사항으로 놓았다는 점이다.

### 핵심 기여
- 7B 규모 음성 인코더(w2v-BERT 계열 자기지도)와 LLM 스타일 디코더를 결합해, 언어 ID가 학습 시 고정되지 않는 개방형(open-vocabulary) 다국어 ASR 달성
- In-context learning 방식의 언어 확장 — 몇 개의 (오디오, 전사) 예시만 프롬프트로 주면 학습 없이 새 언어를 인식하는 퓨샷 능력 실증
- 언어 커뮤니티와 협업해 수집한 데이터셋(Omnilingual ASR Corpus)과 모델 가중치를 모두 오픈소스로 공개, 저자원 언어의 데이터 윤리 문제를 정면으로 다룸

### 이 논문이 중요한 이유
"Whisper 이후 ASR은 무엇을 향해 가는가"에 대한 현재 시점의 가장 선명한 답이다. Whisper가 규모(scale)로, USM이 자기지도 사전학습으로 문제를 풀었다면, Omnilingual ASR은 **확장 가능성 자체를 아키텍처 목표로 삼는다**. 프로덕트 관점에서 이 전환은 결정적이다 — 신규 언어·방언·도메인 지원의 한계비용이 "재학습 사이클"에서 "몇 개의 예시"로 떨어지면, 글로벌 음성 제품의 시장 진입 전략이 완전히 달라진다.

### 사전 지식
- Whisper의 대규모 약지도 학습 방식과 그 한계(언어 ID 고정, 롱테일 언어 미지원)
- 인코더-디코더 구조에서 LLM 디코더를 쓰는 음성-언어 모델(SpeechLM) 개념
- In-context learning / 퓨샷 프롬프팅이 텍스트 LLM에서 작동하는 원리
- 저자원 언어 데이터 수집의 윤리적 쟁점(동의, 커뮤니티 소유권, 보상)

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Scaling Speech Technology to 1,000+ Languages / MMS (Pratap et al., 2023)](https://arxiv.org/abs/2305.13516)
- [Google USM: Scaling ASR Beyond 100 Languages (Zhang et al., 2023)](https://arxiv.org/abs/2303.01037)
- [ASR Leaderboard: Reproducible and Transparent Multilingual and Long-Form ASR Evaluation (2025)](https://arxiv.org/abs/2510.06961)

### 실무 적용
AI 더빙·자막·음성 에이전트 제품에서 "지원 언어 수"는 곧 TAM(시장 규모)이다. Omnilingual ASR은 지원 언어 확장을 데이터 수집 + 퓨샷 등록 문제로 바꿔 놓으므로, 신흥 시장이나 소수 언어권 진입 시 ROI 계산이 근본적으로 달라진다. 실무 적용 시에는 7B 모델의 추론 비용이 관건이므로, 고자원 언어는 경량 모델(Parakeet, Whisper-turbo)로 라우팅하고 롱테일 언어만 Omnilingual로 폴백하는 **2-티어 라우팅 아키텍처**가 현실적인 설계다.

---

## 추천 읽기 순서

1. **Paper 1 (Google USM)** — 먼저 읽는다. Whisper라는 익숙한 기준점과 대비되는 "자기지도 + 텍스트 주입" 레시피를 보며, ASR 스케일링에 축이 두 개 있다는 것을 먼저 이해한다.
2. **Paper 2 (data2vec)** — 그다음 한 단계 아래로 내려간다. USM이 의존하는 "좋은 음성 표현이란 무엇인가"라는 표현 학습의 원리를 본다. 3장(방법론)과 4.1절(음성 실험)에 집중하면 충분하다.
3. **Paper 3 (Omnilingual ASR)** — 마지막에 읽는다. 앞의 두 흐름이 2025년 시점에 어떻게 합류해, 확장성이라는 새로운 목표로 재정의되는지를 확인한다.

## 핵심 테이크어웨이

- **ASR의 병목은 모델이 아니라 라벨이다.** 세 논문 모두 "라벨된 음성을 어떻게 덜 쓸 것인가"에 대한 서로 다른 답이다 — 텍스트 주입(USM), 모달리티 독립 자기지도(data2vec), 퓨샷 언어 확장(Omnilingual).
- **표현 학습이 곧 제품 확장성이다.** 좋은 사전학습 인코더는 ASR뿐 아니라 감정 분석, 화자 분리, 음성 검색까지 한 백본으로 커버한다. 인코더 선택은 모델 결정이 아니라 아키텍처 결정이다.
- **"새 언어/도메인 추가 비용"을 제품 지표로 삼아라.** 재학습 사이클(주 단위) → 퓨샷 등록(시간 단위)으로 한계비용이 떨어지는 방향이 지난 3년간 ASR 연구의 일관된 벡터였다.
- **평가는 WER 하나로 끝나지 않는다.** 롱테일 언어, 롱폼 오디오, 코드 스위칭, 노이즈 환경에서의 성능 편차가 실제 사용자 경험을 가른다. ASR Leaderboard 같은 재현 가능한 평가 체계를 내부에도 구축해야 한다.

## 다음 토픽과의 연결

다음 토픽은 **Neural Text-to-Speech**다. 오늘 본 세 논문은 모두 "음성 → 이산/연속 표현 → 텍스트"의 인식(understanding) 방향이었다. TTS는 정확히 그 화살표를 뒤집어 "텍스트 → 표현 → 음성"을 다룬다. 특히 오늘 data2vec에서 본 **표현 학습의 문제의식(이산 단위 vs 연속 잠재)** 은 TTS에서 뉴럴 코덱 토큰(EnCodec, SoundStream) 설계로 그대로 이어진다. 인식과 합성이 같은 표현 공간을 공유하기 시작하는 지점이 바로 음성 파운데이션 모델의 현재 최전선이다.
