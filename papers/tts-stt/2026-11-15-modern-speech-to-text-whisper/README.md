# Daily AI Paper Recommendations

> **Date:** 2026-11-15
> **Module:** Module 5: TTS and STT Model Development
> **Topic:** Modern Speech-to-Text Whisper and Beyond

---

## Paper 1 (Classic): Pushing the Limits of Semi-Supervised Learning for Automatic Speech Recognition
- **Authors:** Yu Zhang, James Qin, Daniel S. Park, Wei Han, Chung-Cheng Chiu, Ruoming Pang, Quoc V. Le, Yonghui Wu
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2010.10504
- **PDF:** [./pushing-limits-semi-supervised-asr-zhang-2020.pdf](./pushing-limits-semi-supervised-asr-zhang-2020.pdf)
- **Citation Count:** ~500+

### 요약
Google Research가 wav2vec 2.0 자기지도 사전학습, Noisy Student 자기학습(self-training), SpecAugment, 대형 Conformer를 하나의 레시피로 결합한 논문입니다. Libri-Light의 6만 시간 무라벨 오디오를 활용해 LibriSpeech test/test-other에서 WER 1.4%/2.6%를 달성하며 당시 SOTA(1.7%/3.3%)를 크게 경신했습니다.

### 핵심 기여
- **사전학습 + 자기학습의 결합**: 두 방법이 서로 대체재가 아니라 보완재임을 실증 — 사전학습으로 초기화한 뒤 의사 라벨(pseudo-label)로 반복 자기학습하면 효과가 누적됨
- **Conformer 스케일링**: 최대 1B 파라미터급 Conformer(XL/XXL)로 확장했을 때 사전학습이 있어야만 대형 모델의 이득이 나타남을 보여줌
- **적응형 SpecAugment 기반 Noisy Student 파이프라인**: 교사 모델이 라벨을 만들고 학생 모델은 강한 노이즈(증강) 속에서 학습하는 구조를 ASR에 맞게 정립

### 이 논문이 중요한 이유
Whisper가 "대량의 약지도(weak supervision) 라벨"로 문제를 풀었다면, 이 논문은 "라벨 없는 대량 오디오"를 최대한 활용하는 정반대 축의 정점입니다. 이후 Google USM, BEST-RQ, 그리고 대부분의 상용 ASR 팀이 쓰는 "사전학습 → 의사 라벨링 → 증류" 파이프라인의 원형이 바로 이 논문입니다. 데이터 전략이 모델 구조만큼 중요하다는 점을 수치로 증명했다는 점에서 AI 엔지니어 필독입니다.

### 사전 지식
- wav2vec 2.0의 대조 학습(contrastive learning)과 양자화 타깃 개념
- Conformer 구조(Self-Attention + Convolution 모듈)
- SpecAugment(시간/주파수 마스킹), Noisy Student 학습(Xie et al., 2020)
- WER(단어 오류율) 지표

### 관련 논문
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477)
- [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)
- [Improved Noisy Student Training for Automatic Speech Recognition (Park et al., 2020)](https://arxiv.org/abs/2005.09629)
- [Self-training with Noisy Student improves ImageNet classification (Xie et al., 2019)](https://arxiv.org/abs/1911.04252)

### 실무 적용
도메인 특화 ASR(콜센터, 의료, 더빙 원본 전사 등)을 만들 때 라벨링 비용이 병목입니다. 이 논문의 레시피를 그대로 적용하면 ① 공개 사전학습 모델로 시작 → ② 소량 라벨로 파인튜닝 → ③ 사내 무라벨 오디오에 의사 라벨 생성 → ④ 신뢰도 필터링 후 재학습을 반복하는 데이터 플라이휠을 구축할 수 있습니다. 실제로 많은 팀이 Whisper-large로 의사 라벨을 만들고 작은 모델을 학습시키는 방식(Distil-Whisper 등)도 이 사고의 연장선입니다.

---

## Paper 2 (Classic): Zipformer: A Faster and Better Encoder for Automatic Speech Recognition
- **Authors:** Zengwei Yao, Liyong Guo, Xiaoyu Yang, Wei Kang, Fangjun Kuang, Yifan Yang, Zengrui Jin, Long Lin, Daniel Povey
- **Year:** 2023 (ICLR 2024)
- **arXiv:** https://arxiv.org/abs/2310.11230
- **PDF:** [./zipformer-yao-2023.pdf](./zipformer-yao-2023.pdf)
- **Citation Count:** ~200+

### 요약
Kaldi의 창시자 Daniel Povey 팀(Xiaomi)이 Conformer를 대체하기 위해 설계한 ASR 인코더입니다. 중간 레이어가 낮은 프레임레이트로 동작하는 U-Net형 구조, 어텐션 가중치 재사용, BiasNorm, SwooshR/SwooshL 활성화 함수, ScaledAdam 옵티마이저를 도입해 LibriSpeech·Aishell-1·WenetSpeech에서 SOTA를 달성하면서 추론 속도는 50% 이상 빠르고 메모리도 적게 씁니다.

### 핵심 기여
- **다중 해상도(U-Net형) 인코더**: 50Hz → 25 → 12.5 → 6.25Hz로 다운샘플한 뒤 다시 올리는 구조로, 연산량을 크게 줄이면서 장거리 문맥 포착
- **어텐션 가중치 재사용**: 한 블록 안에서 계산한 어텐션 가중치를 여러 모듈이 공유해 비용 절감
- **BiasNorm, Swoosh 활성화, ScaledAdam**: LayerNorm의 정보 손실 문제와 Adam의 파라미터 스케일 문제를 해결하는 학습 안정화 기법

### 이 논문이 중요한 이유
Whisper 이후 "큰 모델이 정답"이라는 흐름 속에서, Zipformer는 **동일 정확도를 훨씬 적은 연산으로** 얻는 아키텍처 효율성의 대표 사례입니다. 오픈소스 k2/icefall/sherpa-onnx 생태계의 기본 인코더로 자리잡아 온디바이스·스트리밍 ASR의 사실상 표준이 되었습니다. 모델 구조 설계가 비용과 지연시간(latency)에 어떤 영향을 주는지 배우기에 최적입니다.

### 사전 지식
- Conformer 구조와 RNN-Transducer(RNN-T) 손실
- LayerNorm/정규화 기법, Adam 옵티마이저 동작 원리
- 프레임레이트(다운샘플링)와 시퀀스 길이 · 연산량 관계

### 관련 논문
- [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)
- [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition (Kim et al., 2022)](https://arxiv.org/abs/2206.00888)
- [Sequence Transduction with Recurrent Neural Networks (Graves, 2012)](https://arxiv.org/abs/1211.3711)
- [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (Rekesh et al., 2023)](https://arxiv.org/abs/2305.05084)

### 실무 적용
실시간 자막, 음성 비서, 회의록처럼 **지연시간과 서버 비용이 중요한** 서비스에서 Whisper 대신 스트리밍 Zipformer-Transducer를 sherpa-onnx로 배포하는 사례가 많습니다. 모바일·엣지 배포, 한국어 등 특정 언어 전용 경량 모델 학습 시 1순위 후보이며, "Whisper로 의사 라벨 → Zipformer로 경량 학습" 조합이 비용 대비 효과가 매우 좋습니다.

---

## Paper 3 (Recent): Fun-ASR Technical Report
- **Authors:** Tongyi Fun Team (Alibaba Group)
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2509.12508
- **PDF:** [./fun-asr-technical-report-tongyi-2025.pdf](./fun-asr-technical-report-tongyi-2025.pdf)
- **Citation Count:** 신규 논문 (2025.09 공개, v4 2025.12)

### 요약
Alibaba Tongyi 팀이 공개한 LLM 기반 대규모 ASR 시스템입니다. 0.7B 오디오 인코더 + 7B LLM 디코더(경량판 Fun-ASR-nano는 0.2B + 0.6B)로 구성되며, 대규모 데이터·모델 스케일링·LLM 통합에 **강화학습(RL)** 을 결합했습니다. 특히 "LLM 기반 ASR은 공개 벤치마크에선 강하지만 실제 산업 데이터에선 약하다"는 문제를 정면으로 다루며, 스트리밍·노이즈 강건성·코드스위칭·핫워드 커스터마이징 등 프로덕션 요구사항에 최적화했습니다.

### 핵심 기여
- **LLM 디코더의 환각(hallucination) 억제**: LLM이 들리지 않은 말을 만들어내는 문제를 데이터 구성과 RL로 완화
- **ASR 전용 강화학습 단계 도입**: 인식 정확도와 실사용 품질을 보상으로 최적화하는 후처리 학습
- **프로덕션 기능 내재화**: 스트리밍 추론, 핫워드(고유명사) 주입, 중-영 코드스위칭, 잡음 환경 대응
- **오픈소스 공개**: 코드와 모델을 GitHub(FunAudioLLM/Fun-ASR)로 공개

### 이 논문이 중요한 이유
Whisper(인코더-디코더) → Seed-ASR/Qwen-Audio(LLM 결합)로 이어진 흐름이 "벤치마크 SOTA"에서 "현장 품질"로 관심이 옮겨가고 있음을 잘 보여줍니다. 공개 벤치마크 점수와 실제 서비스 품질 사이의 간극을 명시적으로 측정했다는 점이 PM·엔지니어 모두에게 큰 시사점입니다. 또한 LLM 학습에서 쓰이던 RL을 ASR에 적용한 사례로, 음성 모델에도 "사후 정렬(post-training)" 패러다임이 들어오고 있음을 확인할 수 있습니다.

### 사전 지식
- Whisper 구조와 약지도 대규모 학습 개념
- Speech-LLM 구조(오디오 인코더 + 어댑터 + LLM 디코더)
- RLHF/GRPO 등 강화학습 기반 사후 학습 기초
- 스트리밍 ASR의 청크 단위 처리 개념

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Seed-ASR: Understanding Diverse Speech and Contexts with LLM-based Speech Recognition (Bai et al., 2024)](https://arxiv.org/abs/2407.04675)
- [Qwen2-Audio Technical Report (Chu et al., 2024)](https://arxiv.org/abs/2407.10759)
- [FunAudioLLM: Voice Understanding and Generation Foundation Models (Tongyi SpeechTeam, 2024)](https://arxiv.org/abs/2407.04051)

### 실무 적용
B2B 음성 서비스(콜센터 분석, 회의록, AI 더빙 원본 전사)에서 가장 많이 받는 불만은 **고유명사 오인식**과 **없는 말이 생기는 환각**입니다. Fun-ASR의 핫워드 주입과 환각 억제 전략은 고객사별 용어집을 반영하는 기능 설계에 직접 참고할 수 있습니다. 또한 "공개 벤치마크 vs 사내 실데이터 평가셋" 분리 원칙은 모델 선정 시 반드시 도입해야 할 평가 프로세스입니다.

---

## 추천 읽기 순서
1. **Pushing the Limits (2020)** — 데이터 전략(사전학습 + 자기학습)의 기본기를 먼저 이해합니다.
2. **Zipformer (2023)** — 같은 데이터로 더 싸고 빠르게 만드는 아키텍처 효율성 관점을 익힙니다.
3. **Fun-ASR (2025)** — 데이터 + 모델 + LLM + RL이 결합된 최신 프로덕션 시스템으로 전체 그림을 완성합니다.

## 핵심 테이크어웨이
- **Q. ASR 성능을 끌어올리는 가장 큰 레버는?** → 모델 구조보다 **데이터 전략**. 무라벨 데이터(자기지도·자기학습)든 약라벨 데이터(Whisper)든, 데이터를 어떻게 쓰느냐가 성능의 상한을 결정합니다.
- **Q. 큰 모델만이 답인가?** → 아닙니다. Zipformer처럼 다중 해상도·가중치 재사용으로 **같은 정확도를 절반 비용으로** 얻을 수 있으며, 실시간 서비스에선 이것이 곧 단위 경제성입니다.
- **Q. 벤치마크 SOTA면 서비스에 바로 써도 되나?** → Fun-ASR이 보여주듯 공개 벤치마크와 실서비스 품질은 다릅니다. **자체 도메인 평가셋**과 환각·고유명사 지표를 반드시 따로 측정해야 합니다.
- **Q. 2025년 ASR의 방향은?** → 인코더 + LLM 디코더 구조에 **RL 기반 사후 학습**을 얹어 실사용 품질(환각, 핫워드, 코드스위칭)을 최적화하는 방향입니다.

## 다음 토픽과의 연결
다음 토픽은 **Neural Text-to-Speech**입니다. 오늘 다룬 ASR은 "음성 → 텍스트"였다면, 내일은 그 역방향인 "텍스트 → 음성"입니다. 흥미롭게도 Fun-ASR을 만든 Tongyi 팀은 CosyVoice(TTS)도 만들었고, 대규모 사전학습·LLM 결합·RL 정렬이라는 동일한 패러다임이 TTS에도 그대로 적용되고 있습니다. 또한 고품질 ASR은 TTS 학습 데이터 전사와 생성 음성의 품질 평가(WER 기반 명료도 측정)에 필수 도구이므로, 두 분야가 어떻게 맞물리는지 주목해 보세요.
