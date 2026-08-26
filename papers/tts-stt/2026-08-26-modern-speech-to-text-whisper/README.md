# Daily AI Paper Recommendations

> **Date:** 2026-08-26
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Modern Speech-to-Text Whisper and Beyond

---

## Paper 1 (Classic): Unsupervised Cross-lingual Representation Learning for Speech Recognition (XLSR)
- **Authors:** Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, Michael Auli
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2006.13979
- **PDF:** [./xlsr-conneau-2020.pdf](./xlsr-conneau-2020.pdf)
- **Citation Count:** 약 1,900회

### 요약
XLSR은 wav2vec 2.0의 대조 학습(contrastive learning) 구조를 여러 언어의 raw waveform에 동시에 적용해, 언어를 가로지르는 공통 음성 표현을 사전학습한 모델이다. 53개 언어로 학습된 XLSR-53은 CommonVoice에서 음소 오류율을 상대 72%, BABEL에서 단어 오류율을 상대 16% 개선했다. 핵심 발견은 latent discrete 표현이 언어 간에 실제로 공유되며, 유사한 언어일수록 공유 비율이 높아진다는 점이다.

### 핵심 기여
- 다국어 self-supervised 사전학습이 단일 언어 사전학습을 명확히 능가함을 대규모로 입증
- 언어 간에 공유되는 discrete latent codebook을 통해 저자원 언어가 고자원 언어의 표현을 "빌려 쓰는" 메커니즘을 제시
- 하나의 모델로 다국어 ASR을 처리하는 것이 언어별 개별 모델과 경쟁 가능함을 보임
- XLSR-53 공개로 저자원 음성 연구의 진입 장벽을 크게 낮춤

### 이 논문이 중요한 이유
Whisper 이전에 "다국어 음성 모델은 하나로 통합 가능한가"라는 질문에 처음으로 설득력 있게 답한 논문이다. AI 엔지니어 입장에서 XLSR은 사전학습 인코더 → 태스크별 파인튜닝이라는, 오늘날 음성 파이프라인의 표준 레시피를 확립했다. 한국어처럼 라벨 데이터가 영어 대비 절대적으로 부족한 언어에서 실무적으로 가장 먼저 검토하게 되는 선택지이기도 하다.

### 사전 지식
wav2vec 2.0의 구조(CNN feature encoder + Transformer + product quantization), contrastive loss와 masked prediction의 차이, CTC 파인튜닝, 그리고 PER/WER 평가 지표를 알고 있어야 실험 결과 해석이 가능하다.

### 관련 논문
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale (Babu et al., 2021)](https://arxiv.org/abs/2111.09296)
- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)

### 실무 적용
다국어 자막/더빙 서비스에서 언어별로 ASR 모델을 따로 운영하면 배포·모니터링 비용이 언어 수에 비례해 늘어난다. XLSR 계열 인코더 하나를 백본으로 두고 언어별 어댑터나 CTC 헤드만 교체하면 GPU 메모리와 운영 복잡도를 크게 줄일 수 있다. 신규 언어 지원 시에도 수십 시간 수준의 라벨 데이터로 실사용 가능한 품질에 도달할 수 있어, 시장 확장 속도를 좌우하는 요소가 된다.

---

## Paper 2 (Classic): Scaling Speech Technology to 1,000+ Languages (MMS)
- **Authors:** Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2305.13516
- **PDF:** [./mms-pratap-2023.pdf](./mms-pratap-2023.pdf)
- **Citation Count:** 약 900회

### 요약
MMS(Massively Multilingual Speech)는 종교 텍스트 낭독 음성이라는 독특한 데이터 소스를 활용해 1,400개 이상 언어의 음성 코퍼스를 구축하고, 1,107개 언어의 ASR과 1,100개 이상 언어의 TTS, 4,000개 언어의 언어 식별 모델을 만들었다. Whisper 대비 11개 언어에서 WER을 절반 수준으로 낮추면서도 지원 언어 수는 11배 많다. 언어별 adapter를 사용해 하나의 백본에서 언어를 스위칭한다.

### 핵심 기여
- 웹 크롤링이 불가능한 초저자원 언어에 대해 "정렬 가능한 병렬 낭독 데이터"라는 확장 가능한 데이터 수집 전략을 제시
- 언어당 파라미터 증가를 최소화하는 adapter 기반 다국어 ASR 아키텍처
- ASR·TTS·LID를 하나의 사전학습 표현 위에서 통합
- 1,000개 이상 언어 모델과 데이터를 공개해 언어 다양성 연구의 기준선을 재설정

### 이 논문이 중요한 이유
"모델을 크게"가 아니라 "데이터를 어디서 구할 것인가"로 문제를 재정의한 사례다. AI 엔지니어에게 이 논문의 진짜 교훈은 아키텍처가 아니라 데이터 파이프라인 설계에 있다. 또한 adapter 방식은 지원 언어를 늘릴 때 전체 재학습 없이 증분 배포가 가능하게 해주는데, 이는 제품 로드맵 관점에서 매우 중요한 성질이다.

### 사전 지식
wav2vec 2.0 / XLSR 사전학습, adapter 및 PEFT 개념, 강제 정렬(forced alignment)과 CTC 세그멘테이션, 그리고 저자원 언어 평가에서 WER과 CER 중 무엇을 봐야 하는지에 대한 감각이 필요하다.

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages (Zhang et al., 2023)](https://arxiv.org/abs/2303.01037)
- [Unsupervised Cross-lingual Representation Learning for Speech Recognition (Conneau et al., 2020)](https://arxiv.org/abs/2006.13979)

### 실무 적용
글로벌 더빙·자막 제품에서 "지원 언어 수"는 곧바로 TAM(총 시장 규모)이다. MMS의 adapter 구조는 신규 언어를 추가할 때 기존 언어 품질을 회귀시키지 않으면서 배포할 수 있게 해주므로, 언어 확장을 분기 단위 릴리스 아이템으로 관리할 수 있게 만든다. 다만 종교 텍스트 도메인 편향이 있어 일상 대화·회의록 도메인에서는 반드시 자체 도메인 데이터로 파인튜닝 검증을 거쳐야 한다.

---

## Paper 3 (Recent): Voxtral
- **Authors:** Mistral AI (Alexander H. Liu, Andy Ehrenberg, Andy Lo, et al.)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2507.13264
- **PDF:** [./voxtral-mistral-2025.pdf](./voxtral-mistral-2025.pdf)
- **Citation Count:** 약 60회 (2026년 8월 기준, 빠르게 증가 중)

### 요약
Voxtral Mini(Ministral 3B 기반)와 Voxtral Small(Mistral Small 3.1 24B 기반)은 음성과 텍스트를 함께 이해하는 멀티모달 오디오 챗 모델이다. 전사(transcription) 전용 모델이 아니라 오디오를 입력으로 받는 LLM으로, 32K 컨텍스트를 통해 최대 40분 길이 오디오와 멀티턴 대화를 처리한다. 오디오 입력 상태에서의 function calling을 네이티브로 지원하며 Apache 2.0으로 공개되었다.

### 핵심 기여
- ASR 성능을 유지하면서 텍스트 능력 저하를 최소화하는 오디오-텍스트 통합 학습 레시피
- 32K 컨텍스트 기반 장문 오디오 이해 — 청크 분할 후 재조합이라는 기존 파이프라인의 구조적 한계를 제거
- 오디오 입력에서 바로 도구 호출로 이어지는 네이티브 function calling
- 음성 이해·지식·추론을 측정하는 3종 벤치마크 신규 공개

### 이 논문이 중요한 이유
STT의 정의가 "오디오 → 텍스트"에서 "오디오 → 행동"으로 넘어가는 지점을 보여준다. Whisper 계열이 전사 정확도를 경쟁했다면, Voxtral은 전사·요약·질의응답·도구 호출을 하나의 모델에서 처리한다. Agentic AI 제품을 설계하는 입장에서는 ASR을 별도 마이크로서비스로 둘 것인가, 아니면 오디오를 그대로 에이전트 입력으로 넣을 것인가라는 아키텍처 분기점을 만든다.

### 사전 지식
Whisper 인코더-디코더 구조, LLM에 모달리티를 붙이는 방식(어댑터/프로젝터, cross-attention), function calling과 tool schema, 그리고 오디오 토큰이 컨텍스트 예산을 얼마나 소비하는지에 대한 계산 감각이 필요하다.

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Qwen2-Audio Technical Report (Chu et al., 2024)](https://arxiv.org/abs/2407.10759)
- [Step-Audio 2 Technical Report (2025)](https://arxiv.org/abs/2507.16632)

### 실무 적용
회의록·고객 상담 분석 제품에서 기존에는 ASR → 화자분리 → 요약 LLM → 액션 아이템 추출로 4단계 파이프라인을 운영했다. Voxtral 같은 오디오 LLM은 이 체인을 1~2단계로 압축해 지연시간과 단계별 오류 누적을 줄인다. 단, 전사 정확도만 놓고 보면 특화 ASR 모델이 여전히 우위인 구간이 있으므로, "정확한 축어록이 산출물"인 제품과 "이해와 액션이 산출물"인 제품을 나눠 모델을 선택하는 것이 현실적이다.

---

## 추천 읽기 순서
1. **XLSR (2020)** — 다국어 음성 표현이 왜 하나의 모델로 통합 가능한지, 그 원리를 먼저 이해한다.
2. **MMS (2023)** — 그 원리를 1,000개 언어 규모로 밀어붙였을 때 병목이 모델이 아니라 데이터임을 확인한다.
3. **Voxtral (2025)** — STT가 LLM에 흡수되면서 문제 정의 자체가 어떻게 바뀌는지 본다.

## 핵심 테이크어웨이
- **표현 공유가 저자원 문제의 핵심 해법이다.** 언어별 모델을 늘리는 것보다 공통 표현 + 얇은 언어별 레이어가 품질과 운영 비용 양쪽에서 유리하다.
- **성능 병목은 아키텍처에서 데이터로 이동했다.** MMS의 기여 대부분은 모델이 아니라 "어디서 정렬 가능한 음성을 구할 것인가"라는 데이터 소싱 설계에 있다.
- **어댑터는 제품 로드맵의 문제다.** 증분 배포가 가능한 구조를 고르면 언어 확장이 재학습 프로젝트가 아니라 릴리스 아이템이 된다.
- **STT는 독립 컴포넌트에서 에이전트 입력 모달리티로 재정의되고 있다.** 파이프라인 단축은 지연시간뿐 아니라 오류 누적 구조를 바꾼다.
- **벤치마크 WER은 제품 지표가 아니다.** 도메인 편향(MMS의 종교 텍스트, Whisper의 웹 오디오)이 실사용 품질을 결정하므로 자체 도메인 평가셋이 반드시 필요하다.

## 다음 토픽과의 연결
다음은 **Neural Text-to-Speech**다. 오늘 다룬 세 논문은 모두 "음성을 어떤 표현으로 압축할 것인가"를 다뤘는데, TTS는 정확히 그 반대 방향 — 표현에서 파형을 복원하는 문제다. 특히 MMS가 ASR과 TTS를 같은 사전학습 표현 위에 올린 점, Voxtral이 오디오를 LLM 토큰 공간에 매핑한 점은 이후 나올 codec 기반 TTS(VALL-E 계열)의 discrete token 접근과 직접 이어진다. ASR과 TTS가 별개 기술이 아니라 같은 표현 공간의 인코더/디코더라는 관점을 갖고 다음 논문들을 읽으면 좋다.
