# Daily AI Paper Recommendations

> **Date:** 2026-09-23
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Neural Text-to-Speech

---

## Paper 1 (Classic): FastSpeech: Fast, Robust and Controllable Text to Speech
- **Authors:** Yi Ren, Yangjun Ruan, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu
- **Year:** 2019 (NeurIPS 2019)
- **arXiv:** https://arxiv.org/abs/1905.09263
- **PDF:** [./fastspeech-ren-2019.pdf](./fastspeech-ren-2019.pdf)
- **Citation Count:** ~2,000+

### 요약
Tacotron 계열의 자기회귀(autoregressive) TTS는 느리고, 단어를 건너뛰거나 반복하는 문제가 있었습니다. FastSpeech는 Transformer 기반 피드포워드 네트워크로 멜 스펙트로그램을 **병렬로 한 번에** 생성하고, 교사(teacher) 모델의 어텐션 정렬에서 뽑은 음소 길이(duration)로 "Length Regulator"가 음소 시퀀스를 늘려 프레임 길이를 맞춥니다. 결과적으로 멜 생성 270배, 전체 합성 38배 빨라졌습니다.

### 핵심 기여
- **Non-autoregressive TTS의 표준 정립:** 병렬 생성으로 추론 속도를 수백 배 개선
- **Length Regulator + Duration Predictor:** 텍스트-음성 길이 불일치를 명시적 duration으로 해결 → 단어 누락/반복 거의 제거
- **제어 가능성:** duration 배율만 조정하면 말하기 속도(0.5x~1.5x)와 쉼(break)을 부드럽게 제어

### 이 논문이 중요한 이유
오늘날 실서비스 TTS의 "duration 모델링" 사고방식이 여기서 시작됐습니다. FastSpeech 2, FastPitch, Glow-TTS, VITS 모두 "명시적 길이 예측 → 병렬 생성" 패러다임을 계승합니다. 또한 LLM 기반 TTS가 다시 자기회귀로 돌아간 지금, **"왜 비자기회귀가 등장했는가(속도·안정성)"**를 이해해야 트레이드오프를 판단할 수 있습니다.

### 사전 지식
- Transformer의 self-attention, 인코더-디코더 구조
- 멜 스펙트로그램과 보코더(WaveGlow/HiFi-GAN)의 역할 분리
- 지식 증류(Knowledge Distillation) 개념 — 교사 모델 출력으로 학생 모델 학습

### 관련 논문
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)](https://arxiv.org/abs/2006.04558)
- [Neural Speech Synthesis with Transformer Network (Li et al., 2018)](https://arxiv.org/abs/1809.08895)
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions / Tacotron 2 (Shen et al., 2017)](https://arxiv.org/abs/1712.05884)

### 실무 적용
- **AI 더빙:** 원본 영상 길이에 맞춰 번역 음성을 맞춰야 할 때, duration 배율 제어가 립싱크/타임라인 싱크의 핵심 레버가 됩니다.
- **저지연 음성 에이전트:** 콜센터·IVR처럼 응답 지연이 중요한 곳에서 병렬 생성 구조가 여전히 유리합니다.
- **온디바이스 TTS:** 가볍고 안정적이어서 모바일/임베디드 TTS의 기본 백본으로 널리 쓰입니다.

---

## Paper 2 (Classic): NaturalSpeech: End-to-End Text to Speech Synthesis with Human-Level Quality
- **Authors:** Xu Tan, Jiawei Chen, Haohe Liu, Jian Cong, Chen Zhang, Yanqing Liu, Xi Wang, Yichong Leng, Yuanhao Yi, Lei He, Frank Soong, Tao Qin, Sheng Zhao, Tie-Yan Liu
- **Year:** 2022
- **arXiv:** https://arxiv.org/abs/2205.04421
- **PDF:** [./naturalspeech-tan-2022.pdf](./naturalspeech-tan-2022.pdf)
- **Citation Count:** ~300+

### 요약
"사람 수준의 음질"을 통계적 유의성(MOS/CMOS + Wilcoxon 검정)으로 **정의**하고, 이를 달성한 최초의 TTS 시스템입니다. VAE 기반으로 텍스트에서 파형까지 end-to-end로 생성하며, 음소 사전학습, 미분 가능한 duration 모델링, 양방향 prior/posterior, 메모리 기반 VAE로 텍스트 prior와 음성 posterior 간 격차를 줄였습니다. LJSpeech에서 실제 녹음 대비 CMOS −0.01(p ≫ 0.05)을 기록했습니다.

### 핵심 기여
- **"Human-level" 평가 기준 정립:** 주관 평가를 통계 검정으로 판단하는 가이드라인 제시
- **VAE end-to-end 파이프라인 고도화:** VITS 계열 구조의 prior 용량↑, posterior 복잡도↓를 위한 4가지 모듈 설계
- **Differentiable Duration:** duration을 미분 가능하게 만들어 end-to-end 최적화 가능

### 이 논문이 중요한 이유
TTS 연구의 목표를 "더 좋아졌다"에서 **"사람과 구분 불가능한가"라는 검증 가능한 질문**으로 바꿨습니다. 제품 관점에서 품질 목표를 어떻게 정의하고 측정할지(평가 설계)를 배우기에 좋은 논문이며, 이후 NaturalSpeech 2/3로 이어지는 zero-shot 계열의 출발점입니다.

### 사전 지식
- VAE(변분 오토인코더)와 ELBO, normalizing flow 기초
- VITS 구조 (Kim et al., 2021)
- MOS/CMOS 주관 평가 방법과 통계 검정(Wilcoxon signed-rank)

### 관련 논문
- [Conditional Variational Autoencoder with Adversarial Learning for End-to-End TTS / VITS (Kim et al., 2021)](https://arxiv.org/abs/2106.06103)
- [NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers (Shen et al., 2023)](https://arxiv.org/abs/2304.09116)
- [NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models (Ju et al., 2024)](https://arxiv.org/abs/2403.03100)

### 실무 적용
- **품질 게이트 설계:** 신규 음성 모델 출시 전 CMOS + 통계 검정을 릴리즈 기준으로 삼는 방식에 그대로 활용 가능
- **단일 화자 프리미엄 보이스:** 오디오북·브랜드 보이스처럼 한 화자의 최고 품질이 중요한 경우 VAE end-to-end 구조가 여전히 경쟁력 있음
- **A/B 평가 운영:** 모델 교체 시 "사람이 차이를 느끼는가"를 기준으로 비용 대비 품질 의사결정

---

## Paper 3 (Recent): Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis
- **Authors:** Zhen Ye, Xinfa Zhu, Chi-Min Chan, Xinsheng Wang, Xu Tan, et al. (HKUST, Wei Xue 교신)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2502.04128
- **PDF:** [./llasa-ye-2025.pdf](./llasa-ye-2025.pdf)
- **Citation Count:** ~100+ (빠르게 증가 중)

### 요약
텍스트 LLM에서 검증된 "학습 시 컴퓨트 스케일링"과 "추론 시 컴퓨트 스케일링(o1식)"을 TTS에 적용한 연구입니다. 단일 레이어 VQ 코덱(X-codec2)과 **표준 Llama Transformer 하나**로 TTS를 구성해 LLM 생태계와 완전히 정렬했습니다. 학습 컴퓨트를 늘리면 자연스러움과 운율이 좋아지고, 추론 시 음성 이해 모델을 verifier로 써서 여러 후보 중 탐색하면 감정 표현·음색 일관성·내용 정확도가 향상됩니다. 1B/3B/8B 체크포인트와 학습 코드를 공개했습니다.

### 핵심 기여
- **"TTS = 그냥 LLM" 단순화:** 다단계 코덱/복수 모델 없이 단일 코드북 + 단일 Llama로 구성
- **Train-time 스케일링 법칙 검증:** 모델·데이터 크기 증가가 운율과 텍스트 의미 이해를 개선
- **Inference-time 스케일링:** ASR/화자 유사도/감정 인식 모델을 verifier로 활용한 best-of-N·빔 서치로 품질 향상

### 이 논문이 중요한 이유
FastSpeech가 "속도와 안정성을 위해 자기회귀를 버렸다"면, Llasa는 **"LLM 스케일링의 힘을 얻기 위해 자기회귀로 돌아간"** 흐름의 대표작입니다. 또한 "추론 시 컴퓨트를 더 쓰면 품질이 오른다"는 트레이드오프를 음성에 보여줘, 비용-품질 곡선을 제품 레벨에서 설계할 수 있게 해줍니다.

### 사전 지식
- Llama 등 decoder-only LLM의 next-token prediction
- 신경 오디오 코덱(EnCodec, DAC)과 벡터 양자화(VQ/RVQ)
- Test-time compute / best-of-N / verifier 개념 (o1, PRM 등)

### 관련 논문
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers / VALL-E (Wang et al., 2023)](https://arxiv.org/abs/2301.02111)
- [CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models (Du et al., 2024)](https://arxiv.org/abs/2412.10117)
- [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with One-Stream Decoupled Speech Tokens (Wang et al., 2025)](https://arxiv.org/abs/2503.01710)

### 실무 적용
- **티어별 품질 과금:** 무료 티어는 1회 샘플링, 프리미엄 티어는 verifier 기반 best-of-N — 추론 컴퓨트를 가격 정책과 연결
- **LLM 인프라 재사용:** vLLM 등 기존 LLM 서빙 스택을 그대로 TTS 서빙에 활용 → 운영 비용 절감
- **에이전트 음성 출력:** 감정·운율이 중요한 AI 아바타/더빙에서 verifier로 "감정 일치도"를 자동 검수

---

## 추천 읽기 순서
1. **FastSpeech** — 병렬 TTS와 duration 모델링의 기본 문제의식부터 잡기
2. **NaturalSpeech** — end-to-end VAE로 "사람 수준" 품질을 어떻게 정의·달성했는지 이해
3. **Llasa** — 패러다임이 다시 LLM 자기회귀 + 스케일링으로 이동한 이유와 방법 확인

## 핵심 테이크어웨이
- **Q. TTS 아키텍처 선택의 핵심 축은?** → 속도/안정성(비자기회귀) vs. 표현력/스케일링(LLM 자기회귀). 제품 요구사항(지연, 품질, 비용)이 선택을 결정합니다.
- **Q. 품질은 어떻게 증명하나?** → NaturalSpeech처럼 CMOS + 통계 검정으로 "사람과 구분 불가"를 검증 가능한 목표로 만드세요.
- **Q. 추론 컴퓨트를 더 쓰는 게 의미 있나?** → Llasa는 verifier 기반 탐색으로 감정·정확도가 오른다는 걸 보여줍니다. 품질-비용 곡선을 제품 전략으로 설계할 수 있습니다.

## 다음 토픽과의 연결
다음 토픽은 **Voice Cloning and Speech Synthesis**입니다. 오늘 본 LLM 기반 TTS(Llasa)는 짧은 참조 음성을 프롬프트로 넣어 zero-shot 음성 복제를 수행하는 구조와 직결됩니다. VALL-E, WaveNet 등으로 이어지며 "짧은 샘플로 어떻게 화자 특성을 재현하는가"를 탐구하게 됩니다.
