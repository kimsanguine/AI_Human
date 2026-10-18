# Daily AI Paper Recommendations

> **Date:** 2026-10-19
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Modern Speech-to-Text Whisper and Beyond

---

## Paper 1 (Classic): WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing
- **Authors:** Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei
- **Year:** 2021 (IEEE JSTSP 2022)
- **arXiv:** https://arxiv.org/abs/2110.13900
- **PDF:** [./wavlm-chen-2021.pdf](./wavlm-chen-2021.pdf)
- **Citation Count:** ~2,000+

### 요약
WavLM은 HuBERT식 마스크 예측에 **노이즈·다른 화자 음성을 섞은 입력에서 원래 발화를 복원하는 denoising 과제**를 더해, ASR뿐 아니라 화자 인식·화자 분리(diarization)·감정 인식까지 하나의 사전학습 모델로 다루도록 설계한 자기지도 음성 모델입니다. 94k시간 규모 데이터로 학습해 SUPERB 벤치마크 전 과제에서 당시 최고 성능을 기록했습니다.

### 핵심 기여
- **Masked speech denoising + prediction:** 겹친 발화/노이즈를 넣고 주 화자의 숨은 단위를 예측하게 해 "누가 말하는지"와 "무엇을 말하는지"를 함께 학습
- **Gated relative position bias:** Transformer에 게이트 상대 위치 편향을 넣어 음성 시퀀스 모델링 성능 향상
- **Full-stack 음성 표현:** 인식(ASR)뿐 아니라 화자·패럴링귀스틱 과제까지 하나의 백본으로 통합 (SUPERB SOTA)

### 이 논문이 중요한 이유
Whisper는 "무엇을 말했는가"에 집중한 지도학습 모델입니다. 반면 실제 서비스에는 화자 분리, 화자 임베딩, 음성 품질 평가 같은 "누가·어떻게"의 문제가 함께 있습니다. WavLM은 이 영역의 사실상 표준 백본으로, 화자 유사도(SIM) 평가 지표(WavLM-TDNN), 화자 분리 파이프라인 등에서 지금도 쓰입니다.

### 사전 지식
- wav2vec 2.0 / HuBERT의 마스크 예측 기반 자기지도학습
- Transformer 위치 인코딩(절대/상대)
- 화자 검증(speaker verification), 화자 분리의 기본 개념

### 관련 논문
- [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)
- [wav2vec 2.0 (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [SUPERB: Speech processing Universal PERformance Benchmark (Yang et al., 2021)](https://arxiv.org/abs/2105.01051)

### 실무 적용
- **AI 더빙/보이스 클로닝 품질 평가:** 원본 화자와 합성 음성의 화자 유사도(SIM)를 WavLM 기반 임베딩으로 측정
- **회의록/콜센터 STT:** 화자 분리 모델의 특징 추출기로 사용해 "누가 말했는지" 정확도 향상
- **노이즈 환경 음성 인식:** denoising 사전학습 덕분에 겹친 발화·잡음에 강건

---

## Paper 2 (Classic): Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling
- **Authors:** Sanchit Gandhi, Patrick von Platen, Alexander M. Rush
- **Year:** 2023
- **arXiv:** https://arxiv.org/abs/2311.00430
- **PDF:** [./distil-whisper-gandhi-2023.pdf](./distil-whisper-gandhi-2023.pdf)
- **Citation Count:** ~300+

### 요약
Distil-Whisper는 Whisper large-v2를 교사로 삼아 **21,170시간 규모의 의사 라벨(pseudo label)** 로 학생 모델을 증류한 연구입니다. 디코더 층을 32개에서 2개로 줄여 **5.8배 빠르고 파라미터는 51% 적으면서**, OOD 테스트에서 WER 차이를 1% 이내로 유지했습니다.

### 핵심 기여
- **WER 휴리스틱 필터링:** 교사 출력과 정답 전사의 WER이 높은 샘플을 버려 환각(hallucination)이 섞인 의사 라벨 제거
- **디코더만 줄이는 구조:** 인코더는 유지하고 디코더를 대폭 축소 — 자기회귀 디코딩이 추론 시간의 병목임을 활용
- **Speculative decoding 보조 모델:** 원본 Whisper의 draft 모델로 써서 출력은 그대로 유지하고 2배 속도 향상
- 긴 오디오에서 환각·반복 오류를 원본보다 줄임

### 이 논문이 중요한 이유
"Whisper가 좋은 건 알겠는데 비싸고 느리다"는 실무의 가장 흔한 문제에 대한 정석 답안입니다. 대형 모델 → 의사 라벨 → 작은 모델로 옮기는 증류 레시피는 STT뿐 아니라 LLM 경량화(Module 6 Day 17)에도 그대로 적용됩니다.

### 사전 지식
- Whisper의 encoder-decoder 구조와 30초 청크 처리 방식
- Knowledge distillation (KL divergence 기반 soft label 학습)
- Speculative decoding 개념

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2022)](https://arxiv.org/abs/2211.17192)

### 실무 적용
- **실시간 자막/라이브 전사:** GPU 비용을 크게 낮추면서 Whisper급 품질 유지
- **온디바이스/엣지 STT:** 경량 모델로 모바일·브라우저 환경 배포
- **도메인 특화 STT:** 자사 오디오를 대형 모델로 의사 라벨링 → 작은 모델로 증류하는 데이터 플라이휠 구축

---

## Paper 3 (Recent): Less is More: Accurate Speech Recognition & Translation without Web-Scale Data (Canary)
- **Authors:** Krishna C. Puvvada, Piotr Żelasko, He Huang, Oleksii Hrinchuk, Nithin Rao Koluguri, Kunal Dhawan, Somshubra Majumdar, Elena Rastorgueva, Zhehuai Chen, Vitaly Lavrukhin, Jagadeesh Balam, Boris Ginsburg
- **Year:** 2024 (Interspeech 2024)
- **arXiv:** https://arxiv.org/abs/2406.19674
- **PDF:** [./canary-less-is-more-puvvada-2024.pdf](./canary-less-is-more-puvvada-2024.pdf)
- **Citation Count:** ~80+

### 요약
NVIDIA의 Canary는 영어·독일어·스페인어·프랑스어 ASR과 음성 번역을 지원하는 모델로, Whisper·OWSM·SeamlessM4T보다 **약 10배 적은 학습 데이터(약 86k시간)** 로 더 높은 정확도를 달성했습니다. "웹 스케일 데이터가 꼭 필요한가?"라는 질문에 아키텍처와 데이터 전략으로 답한 논문입니다.

### 핵심 기여
- **FastConformer 인코더 + Transformer 디코더:** 8배 다운샘플링으로 긴 오디오를 효율적으로 처리
- **MT 기반 합성 번역 데이터:** 기계번역으로 음성 번역 학습 데이터를 생성
- **데이터 밸런싱 & 동적 데이터 블렌딩:** 언어·도메인 비율을 학습 중 조절
- Hugging Face Open ASR Leaderboard 상위권, 모델·코드 공개

### 이 논문이 중요한 이유
Whisper 이후 STT 경쟁의 축이 "데이터 양"에서 **"데이터 품질 + 효율적 아키텍처"** 로 이동했음을 보여줍니다. 대규모 데이터가 없는 팀(특정 언어·도메인)도 경쟁력 있는 모델을 만들 수 있다는 근거가 됩니다.

### 사전 지식
- Conformer 아키텍처 (Gulati et al., 2020)
- 멀티태스크 프롬프트 토큰 방식 (Whisper의 task/language token)
- 음성 번역(AST)의 기본 개념

### 관련 논문
- [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (Rekesh et al., 2023)](https://arxiv.org/abs/2305.05084)
- [OWSM: Reproducing Whisper-Style Training Using an Open-Source Toolkit (Peng et al., 2023)](https://arxiv.org/abs/2309.13876)
- [SeamlessM4T: Massively Multilingual & Multimodal Machine Translation (Seamless Communication et al., 2023)](https://arxiv.org/abs/2308.11596)

### 실무 적용
- **AI 더빙 파이프라인:** ASR + 번역을 하나의 모델로 처리해 STT→MT 단계 오류 누적 감소
- **STT 모델 선택:** Whisper만이 답이 아님 — 지원 언어가 맞으면 Canary/Parakeet 계열이 정확도·속도에서 유리할 수 있음
- **자체 모델 학습 전략:** 데이터 수집보다 데이터 큐레이션·블렌딩에 먼저 투자하는 근거

---

## 추천 읽기 순서
1. **Distil-Whisper** — Whisper를 이미 안다면 가장 빠르게 읽히고, 실무 비용 문제와 바로 연결됩니다.
2. **Canary** — 4페이지 분량. "Whisper보다 적은 데이터로 더 잘하기"의 아키텍처·데이터 전략을 파악합니다.
3. **WavLM** — 인식 너머(화자·노이즈·감정)로 시야를 넓히는 논문. 다음 날 TTS/보이스 클로닝 평가와 직결됩니다.

## 핵심 테이크어웨이
- **Q. Whisper 이후 STT 연구는 어디로 갔나?** → 세 갈래: ① 더 넓은 과제(WavLM: 화자·노이즈), ② 더 싸고 빠르게(Distil-Whisper: 증류), ③ 더 적은 데이터로(Canary: 데이터 품질·효율 아키텍처).
- **Q. 실무에서 가장 먼저 시도할 것은?** → 대형 모델로 의사 라벨 → WER 필터링 → 작은 모델 증류. 품질은 거의 유지하면서 비용을 크게 줄이는 검증된 레시피입니다.
- **Q. 데이터가 적으면 포기해야 하나?** → 아닙니다. Canary는 데이터 밸런싱·합성 데이터·효율적 인코더로 10배 적은 데이터로 SOTA를 냈습니다.

## 다음 토픽과의 연결
다음 주제는 **Neural Text-to-Speech**입니다. 오늘 본 WavLM은 TTS·보이스 클로닝 모델의 화자 유사도 평가에 쓰이고, Distil-Whisper/Canary 같은 STT 모델은 합성 음성의 WER(명료도) 평가에 쓰입니다. 즉 STT는 TTS의 **"귀(평가자)"** 역할을 합니다.
