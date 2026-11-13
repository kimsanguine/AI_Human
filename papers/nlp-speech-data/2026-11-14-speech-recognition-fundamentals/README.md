# Daily AI Paper Recommendations

> **Date:** 2026-11-14
> **Module:** Module 4: NLP and Speech Data
> **Topic:** Speech Recognition Fundamentals

---

## Paper 1 (Classic): EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding
- **Authors:** Yajie Miao, Mohammad Gowayyed, Florian Metze
- **Year:** 2015
- **arXiv:** https://arxiv.org/abs/1507.08240
- **PDF:** [./eesen-miao-2015.pdf](./eesen-miao-2015.pdf)
- **Citation Count:** ~1,000+

### 요약
EESEN은 하나의 깊은 양방향 LSTM이 음소나 문자를 직접 예측하고, CTC로 음성과 라벨의 정렬을 스스로 학습하도록 만든 End-to-End ASR 프레임워크입니다. 여기에 WFST(Weighted Finite-State Transducer) 기반 디코딩을 붙여, 발음 사전과 언어 모델을 CTC 출력에 효율적으로 결합했습니다. 그 결과 기존 하이브리드 DNN-HMM 시스템과 비슷한 WER을 내면서 디코딩 속도는 크게 높였습니다.

### 핵심 기여
- HMM/GMM 정렬 없이 CTC만으로 학습하는 단일 RNN 음향 모델 파이프라인 제시
- CTC 라벨·발음 사전·언어 모델을 T, L, G 세 개의 WFST로 표현하고 하나의 탐색 그래프(TLG)로 합성하는 디코딩 방식 제안
- Kaldi 스타일의 오픈소스 툴킷으로 공개하여 End-to-End ASR의 재현성과 실무 도입 장벽을 낮춤

### 이 논문이 중요한 이유
"End-to-End 모델 + 외부 언어 모델 결합"이라는, 지금도 실무 ASR에서 쓰이는 구조의 출발점입니다. 순수 신경망 모델만으로는 도메인 어휘나 고유명사 처리에 한계가 있는데, EESEN은 전통적인 WFST 디코딩의 장점을 버리지 않고 End-to-End 학습과 합칠 수 있음을 보여줬습니다. CTC 기반 모델과 n-gram LM shallow fusion을 이해하는 기준점이 됩니다.

### 사전 지식
- CTC(Connectionist Temporal Classification) 손실 함수와 blank 토큰 개념
- LSTM/BiLSTM 구조
- HMM 기반 하이브리드 ASR의 기본 파이프라인(음향 모델·발음 사전·언어 모델)
- WFST의 합성(composition) 개념 (가볍게)

### 관련 논문
- [Connectionist Temporal Classification (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)
- [Towards End-to-End Speech Recognition with Recurrent Neural Networks (Graves & Jaitly, 2014)](https://proceedings.mlr.press/v32/graves14.html)
- [Deep Speech: Scaling up end-to-end speech recognition (Hannun et al., 2014)](https://arxiv.org/abs/1412.5567)

### 실무 적용
콜센터·회의록처럼 특정 도메인 용어가 많은 서비스에서는 CTC 음향 모델에 도메인 n-gram LM이나 핫워드(hotword) 그래프를 WFST로 결합하는 방식이 여전히 쓰입니다. NVIDIA NeMo, k2/icefall, WeNet 등의 CTC + WFST/LM 디코더가 EESEN의 설계를 그대로 잇고 있어, 재학습 없이 어휘를 바꾸는 "커스텀 어휘" 기능을 만들 때 직접 응용할 수 있습니다.

---

## Paper 2 (Classic): Jasper: An End-to-End Convolutional Neural Acoustic Model
- **Authors:** Jason Li, Vitaly Lavrukhin, Boris Ginsburg, Ryan Leary, Oleksii Kuchaiev, Jonathan M. Cohen, Huyen Nguyen, Ravi Teja Gadde
- **Year:** 2019
- **arXiv:** https://arxiv.org/abs/1904.03288
- **PDF:** [./jasper-li-2019.pdf](./jasper-li-2019.pdf)
- **Citation Count:** ~700+

### 요약
Jasper는 1D 컨볼루션, 배치 정규화, ReLU, 드롭아웃만으로 이루어진 단순한 End-to-End CTC 음향 모델입니다. 블록 사이에 Dense Residual 연결을 넣어 최대 54층까지 깊게 쌓았고, 새 옵티마이저 NovoGrad도 함께 제안했습니다. LibriSpeech test-clean에서 언어 모델 없이 greedy 디코딩으로 3.86%, Transformer-XL LM과 결합해 2.95% WER을 달성했습니다.

### 핵심 기여
- RNN 없이 순수 컨볼루션만으로 SOTA 수준의 ASR 성능을 낼 수 있음을 입증
- 깊은 컨볼루션 네트워크의 학습을 안정시키는 Dense Residual(DR) 연결 구조 제안
- 레이어별 정규화된 그래디언트를 쓰는 NovoGrad 옵티마이저 제안
- GPU 추론 효율에 최적화된 단순 연산 구성으로 실제 배포 친화적인 설계 제시

### 이 논문이 중요한 이유
RNN은 순차 연산이라 병렬화가 어렵고 추론이 느립니다. Jasper는 "단순한 연산을 깊게 쌓는 것"만으로 정확도와 속도를 동시에 잡을 수 있다는 것을 보여줬습니다. 이 계보가 QuartzNet → Citrinet → FastConformer → Parakeet으로 이어지기 때문에, NVIDIA 계열 오픈소스 ASR 모델을 이해하려면 반드시 거쳐야 하는 논문입니다.

### 사전 지식
- 1D 컨볼루션과 receptive field 개념
- Residual/Dense 연결(ResNet, DenseNet)
- CTC 손실과 greedy/beam search 디코딩
- 배치 정규화와 기본 옵티마이저(SGD, Adam)

### 관련 논문
- [Wav2Letter: an End-to-End ConvNet-based Speech Recognition System (Collobert et al., 2016)](https://arxiv.org/abs/1609.03193)
- [QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions (Kriman et al., 2019)](https://arxiv.org/abs/1910.10261)
- [Conformer: Convolution-augmented Transformer for Speech Recognition (Gulati et al., 2020)](https://arxiv.org/abs/2005.08100)

### 실무 적용
엣지 디바이스나 대량 배치 전사처럼 지연 시간과 GPU 비용이 중요한 환경에서는 Jasper/QuartzNet 계열의 컨볼루션 CTC 모델이 여전히 좋은 선택입니다. TensorRT 등으로 최적화하기 쉬워 실시간 자막, 음성 명령 인식에 적합하며, 더빙·자막 서비스의 1차 전사 엔진으로 쓰고 LLM으로 후처리하는 구성에도 잘 맞습니다.

---

## Paper 3 (Recent): Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST
- **Authors:** Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, Piotr Zelasko, Travis Bartley, Nikolay Karpov, Jagadeesh Balam, Boris Ginsburg
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2509.14128
- **PDF:** [./canary-1b-v2-parakeet-tdt-v3-sekoyan-2025.pdf](./canary-1b-v2-parakeet-tdt-v3-sekoyan-2025.pdf)
- **Citation Count:** 신규 논문 (2025년 9월 공개, Hugging Face Open ASR Leaderboard 상위권 모델)

### 요약
NVIDIA가 공개한 다국어 음성 인식(ASR) 및 음성 번역(AST) 모델 기술 보고서입니다. Canary-1B-v2는 FastConformer 인코더와 Transformer 디코더로 구성되며, Granary 등 총 170만 시간 데이터로 25개 언어(주로 유럽어)를 학습했습니다. Whisper-large-v3보다 영어 ASR 정확도가 높으면서 약 10배 빠르고, 600M 파라미터의 Parakeet-TDT-0.6B-v3도 같은 25개 언어를 지원합니다.

### 핵심 기여
- 사전학습 → 파인튜닝 2단계 학습과 동적 데이터 밸런싱으로 다국어 성능 균형 확보
- 학습 데이터에 비음성(non-speech) 오디오를 섞어 ASR/AST의 환각(hallucination) 감소
- nGPT 인코더와 FastConformer를 비교: nGPT는 대규모 데이터에서 확장성이 좋고, FastConformer는 파인튜닝 후 더 우수함을 실험으로 제시
- 보조 CTC 모델 + NeMo Forced Aligner로 세그먼트 단위 타임스탬프 제공
- TDT(Token-and-Duration Transducer) 디코더를 쓰는 경량 Parakeet-TDT-0.6B-v3 함께 공개

### 이 논문이 중요한 이유
"대형 LLM 기반 음성 모델이 아니어도, 잘 설계된 인코더-디코더/트랜스듀서 모델이 더 빠르고 정확할 수 있다"는 실증 자료입니다. 오늘 고전 두 편(CTC+WFST, 컨볼루션 CTC)의 아이디어가 FastConformer, 보조 CTC 정렬, TDT 디코더로 어떻게 진화했는지 한 번에 볼 수 있습니다. 또한 환각 억제, 타임스탬프, 데이터 밸런싱 등 프로덕션에서 실제로 부딪히는 문제를 다룹니다.

### 사전 지식
- Conformer/FastConformer 인코더 구조
- RNN-T(Transducer) 및 CTC 디코딩의 차이
- Whisper 스타일 멀티태스크 인코더-디코더 모델
- WER, BLEU/COMET 등 ASR·번역 평가 지표

### 관련 논문
- [Robust Speech Recognition via Large-Scale Weak Supervision / Whisper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (Rekesh et al., 2023)](https://arxiv.org/abs/2305.05084)
- [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations / TDT (Xu et al., 2023)](https://arxiv.org/abs/2304.06795)
- [Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual Speech Recognition Evaluation (Srivastav et al., 2025)](https://arxiv.org/abs/2510.06961)

### 실무 적용
AI 더빙·자막 파이프라인에서 "전사 + 번역 + 타임스탬프"를 하나의 모델로 처리할 수 있어 시스템 구성이 단순해집니다. Whisper 대비 10배 빠른 속도는 대용량 영상 처리 비용을 직접 낮추고, 비음성 구간 환각 감소는 음악·효과음이 섞인 콘텐츠에서 품질 이슈를 줄여줍니다. 오픈 웨이트(CC-BY)로 공개되어 자체 호스팅 및 도메인 파인튜닝이 가능합니다.

---

## 추천 읽기 순서
1. **EESEN (2015)** — CTC 기반 End-to-End 모델에 언어 모델을 붙이는 "기본 골격"을 먼저 이해합니다.
2. **Jasper (2019)** — 같은 CTC 프레임 안에서 RNN을 컨볼루션으로 바꿔 속도와 깊이를 확보하는 과정을 봅니다.
3. **Canary-1B-v2 & Parakeet-TDT-v3 (2025)** — 앞의 아이디어가 FastConformer, 트랜스듀서, 다국어·번역까지 확장된 현재 형태를 확인합니다.

## 핵심 테이크어웨이
- **정렬 문제의 해결이 End-to-End ASR의 핵심:** CTC 덕분에 프레임 단위 정렬 라벨 없이 학습할 수 있게 되었고, 이 원리는 2025년 모델의 타임스탬프(보조 CTC)에도 그대로 쓰입니다.
- **외부 지식 결합은 여전히 중요:** EESEN의 WFST 결합은 오늘날 LM fusion, 핫워드 부스팅, 컨텍스트 바이어싱으로 이어집니다.
- **아키텍처는 "병렬화 가능한 단순함"으로 진화:** RNN → 컨볼루션(Jasper) → Conformer/FastConformer로 갈수록 학습·추론 효율이 좋아졌습니다.
- **프로덕션 품질은 데이터 설계에서 나옴:** 데이터 밸런싱과 비음성 데이터 추가처럼 데이터 레시피가 모델 구조만큼 성능을 좌우합니다.

## 다음 토픽과의 연결
다음 토픽은 **Module 5의 "Modern Speech-to-Text: Whisper and Beyond"**입니다. 오늘 살펴본 CTC·컨볼루션·트랜스듀서 기반 모델과 비교해, Whisper처럼 대규모 약지도(weak supervision) 데이터로 학습한 인코더-디코더 모델과 wav2vec 2.0 같은 자기지도 사전학습이 어떻게 범용 음성 인식을 가능하게 했는지 이어서 살펴봅니다. Canary-1B-v2가 Whisper-large-v3를 기준선으로 삼은 이유도 다음 날 더 명확해집니다.
