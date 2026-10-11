# Daily AI Paper Recommendations

> **Date:** 2026-10-12
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** RNN LSTM and Sequence Models

---

## Paper 1 (Classic): Recurrent Neural Network Regularization
- **Authors:** Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1409.2329
- **PDF:** [./rnn-regularization-zaremba-2014.pdf](./rnn-regularization-zaremba-2014.pdf)
- **Citation Count:** ~3,500+

### 요약
Dropout은 피드포워드 신경망에서는 가장 강력한 정규화 기법이었지만, RNN/LSTM에 그대로 적용하면 성능이 오히려 나빠졌습니다. 이 논문은 dropout을 **순환 연결(recurrent connection)에는 적용하지 않고, 층과 층 사이의 비순환 연결에만 적용**하는 간단한 방법으로 이 문제를 해결했습니다. 그 결과 언어 모델링(Penn Treebank), 음성 인식, 이미지 캡셔닝, 기계 번역 전반에서 과적합이 크게 줄었습니다.

### 핵심 기여
- "왜 RNN에 dropout이 안 되는가?"에 대한 답: 시간축으로 노이즈가 누적되어 장기 기억을 망가뜨린다는 점을 지적
- 수직(layer 간) 연결에만 dropout을 거는 정규화 레시피 제시 — 구현이 몇 줄이면 끝남
- Penn Treebank 언어 모델링에서 대형 LSTM으로 당시 SOTA perplexity(약 78)를 달성하며 이후 수년간 표준 베이스라인이 됨

### 이 논문이 중요한 이유
"큰 모델 + 올바른 정규화 = 성능 향상"이라는 공식을 RNN 세계에 처음 정착시킨 논문입니다. 이후 Variational Dropout(Gal & Ghahramani), AWD-LSTM 등 RNN 정규화 연구의 출발점이 되었고, 공개된 코드(PTB LSTM)는 TensorFlow/PyTorch 공식 튜토리얼의 기반이 되었습니다. 저자 중 Sutskever·Vinyals는 바로 직후 Seq2Seq를 발표했는데, 이 정규화 기법이 그 성공의 숨은 기반입니다.

### 사전 지식
- LSTM의 게이트 구조(입력/망각/출력 게이트, 셀 상태)
- Dropout의 기본 개념과 과적합(overfitting)
- 언어 모델 평가 지표인 Perplexity

### 관련 논문
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [A Theoretically Grounded Application of Dropout in RNNs (Gal & Ghahramani, 2016)](https://arxiv.org/abs/1512.05287)
- [Regularizing and Optimizing LSTM Language Models / AWD-LSTM (Merity et al., 2017)](https://arxiv.org/abs/1708.02182)

### 실무 적용
시계열 예측, 온디바이스 음성 인식(키워드 스포팅), 소형 언어 모델처럼 **데이터가 적고 RNN을 쓰는 환경**에서는 지금도 "층 사이에만 dropout" 원칙이 기본값입니다. PyTorch `nn.LSTM(dropout=...)` 파라미터가 정확히 이 논문의 방식(마지막 층 제외, 층 사이에만 적용)으로 동작합니다.

---

## Paper 2 (Classic): LSTM: A Search Space Odyssey
- **Authors:** Klaus Greff, Rupesh Kumar Srivastava, Jan Koutník, Bas R. Steunebrink, Jürgen Schmidhuber
- **Year:** 2015 (IEEE TNNLS 2017)
- **arXiv:** https://arxiv.org/abs/1503.04069
- **PDF:** [./lstm-search-space-odyssey-greff-2015.pdf](./lstm-search-space-odyssey-greff-2015.pdf)
- **Citation Count:** ~6,000+

### 요약
LSTM이 등장한 후 수많은 변형(peephole, coupled gate, GRU 유사 구조 등)이 제안되었지만, "어떤 부품이 정말 중요한가?"를 체계적으로 검증한 연구는 없었습니다. 이 논문은 8가지 LSTM 변형을 음성 인식(TIMIT), 필기 인식(IAM), 다성 음악 모델링(JSB Chorales) 세 과제에서 **총 5,400회 실험**으로 비교했습니다. 결론은 "표준 LSTM을 확실히 이기는 변형은 없으며, 가장 중요한 부품은 망각 게이트(forget gate)와 출력 활성화 함수"였습니다.

### 핵심 기여
- 8가지 LSTM 변형에 대한 최대 규모의 ablation 연구
- 망각 게이트·출력 활성화 제거 시 성능이 크게 하락함을 실증 → LSTM의 "핵심 부품" 식별
- fANOVA로 하이퍼파라미터 중요도 분석: **학습률이 가장 중요**, 하이퍼파라미터 간 상호작용은 작아 독립적으로 튜닝 가능함을 보임

### 이 논문이 중요한 이유
"새 아키텍처를 제안하기 전에, 기존 구조의 무엇이 작동하는지부터 측정하라"는 실증 연구의 모범 사례입니다. 오늘의 Paper 3(minLSTM/minGRU)이 "게이트를 얼마나 덜어낼 수 있나?"를 묻는다면, 이 논문은 그 질문의 10년 전 버전입니다. 하이퍼파라미터 튜닝 우선순위(학습률 → 네트워크 크기 → 모멘텀은 거의 무관)에 대한 실무 지침도 여전히 유효합니다.

### 사전 지식
- LSTM 원본 구조와 peephole connection 개념
- Ablation study와 random search 기반 하이퍼파라미터 탐색
- (선택) fANOVA 같은 하이퍼파라미터 중요도 분석 기법

### 관련 논문
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://doi.org/10.1162/neco.1997.9.8.1735)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (Chung et al., 2014)](https://arxiv.org/abs/1412.3555)
- [An Empirical Exploration of Recurrent Network Architectures (Jozefowicz et al., 2015)](https://proceedings.mlr.press/v37/jozefowicz15.html)

### 실무 적용
시계열·센서·로그 데이터용 RNN을 설계할 때 "이국적인 변형보다 표준 LSTM + 망각 게이트 bias 초기화 + 학습률 튜닝"이 가장 비용 대비 효과가 크다는 근거가 됩니다. 또한 AI 제품팀이 새 모델 구조를 도입할 때 **ablation 기반 의사결정 프로세스**를 설계하는 레퍼런스로 활용할 수 있습니다.

---

## Paper 3 (Recent): Were RNNs All We Needed?
- **Authors:** Leo Feng, Frederick Tung, Mohamed Osama Ahmed, Yoshua Bengio, Hossein Hajimirsadeghi
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2410.01201
- **PDF:** [./were-rnns-all-we-needed-feng-2024.pdf](./were-rnns-all-we-needed-feng-2024.pdf)
- **Citation Count:** ~150+

### 요약
LSTM/GRU가 Transformer에 밀린 가장 큰 이유는 "이전 은닉 상태에 의존하는 게이트" 때문에 학습을 병렬화할 수 없다는 점이었습니다. 이 논문은 게이트에서 이전 상태 의존성(h_{t-1})과 tanh 같은 출력 범위 제약을 제거한 **minLSTM / minGRU**를 제안합니다. 이렇게 하면 parallel scan으로 학습이 가능해져 길이 512 시퀀스에서 약 175배 빨라지고, 파라미터도 크게 줄면서 Mamba 등 최신 SSM과 비슷한 성능을 냅니다.

### 핵심 기여
- 게이트를 입력 x_t에만 의존하게 만들어 LSTM/GRU를 선형 재귀 형태로 변환 → parallel scan 학습 가능
- 기존 대비 파라미터 수를 크게 절감(GRU 대비 최대 ~87%, LSTM 대비 ~85%)
- Selective Copying, 강화학습(D4RL), 문자 단위 언어 모델링에서 Mamba·S6 등과 경쟁력 있는 성능
- 순수 PyTorch 몇 줄로 구현 가능한 단순성

### 이 논문이 중요한 이유
Mamba, RWKV, xLSTM, Griffin 등 2023~2025년 "RNN 르네상스" 모델들의 공통 분모가 무엇인지를 가장 단순한 형태로 보여줍니다. 즉, **핵심은 복잡한 SSM 이론이 아니라 '입력 의존 게이트 + 선형 재귀 + 병렬 스캔'**이라는 통찰입니다. 이 관점을 이해하면 이후 등장하는 새로운 sub-quadratic 아키텍처를 빠르게 해석할 수 있습니다. (단, 대규모 LLM 스케일에서의 검증은 아직 제한적이라는 점은 비판적으로 읽을 부분입니다.)

### 사전 지식
- LSTM/GRU의 게이트 수식 (오늘의 Paper 1, 2)
- Parallel prefix scan(Blelloch scan)의 개념
- Mamba / S4 등 State Space Model의 기본 아이디어

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [Parallelizing Linear Recurrent Neural Nets Over Sequence Length (Martin & Cundy, 2017)](https://arxiv.org/abs/1709.04057)
- [xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)](https://arxiv.org/abs/2405.04517)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)](https://arxiv.org/abs/2006.16236)

### 실무 적용
추론 시 **메모리가 시퀀스 길이와 무관하게 상수**라는 RNN의 장점을 유지하므로, 온디바이스 음성 처리, 실시간 스트리밍(자막·더빙 파이프라인의 VAD/ASR 전처리), 엣지 센서 이상 탐지처럼 지연시간과 메모리가 제약인 환경에 적합합니다. 학습은 Transformer처럼 병렬로, 추론은 RNN처럼 스트리밍으로 — 이 "학습/추론 이중 모드" 패턴은 하이브리드 LLM(Jamba, Hymba 등) 설계의 기본 전제이기도 합니다.

---

## 추천 읽기 순서
1. **Recurrent Neural Network Regularization (2014)** — 8페이지로 짧고, LSTM을 "실제로 잘 학습시키는 법"부터 잡습니다.
2. **LSTM: A Search Space Odyssey (2015)** — LSTM의 어떤 부품이 정말 필요한지 실증적으로 확인합니다.
3. **Were RNNs All We Needed? (2024)** — 2번의 질문("무엇을 덜어낼 수 있나?")을 병렬화 관점에서 극단까지 밀어붙인 현대적 답을 읽습니다.

> 탐구 질문: Paper 2는 "망각 게이트가 가장 중요하다"고 했습니다. Paper 3의 minLSTM은 망각 게이트를 남기되 h_{t-1} 의존성을 제거했습니다. 무엇을 잃고 무엇을 얻었을까요?

## 핵심 테이크어웨이
- **정규화는 구조에 맞게**: 같은 dropout도 어디에 거느냐에 따라 독이 되기도, 약이 되기도 합니다.
- **망각 게이트가 LSTM의 심장**: 10년간의 연구에서 살아남은 부품은 결국 "무엇을 잊을지 정하는 게이트"입니다.
- **병렬화가 아키텍처의 운명을 가른다**: RNN이 밀려난 이유는 표현력보다 학습 병렬성이었고, 그 제약을 풀자 RNN이 다시 경쟁력을 얻었습니다.
- **단순화는 강력한 연구 전략**: 세 논문 모두 "더하기"보다 "빼기/올바르게 배치하기"로 큰 효과를 냈습니다.

## 다음 토픽과의 연결
다음 토픽은 **Optimization and Regularization (Adam, AdamW 등)** 입니다. 오늘 Paper 1의 dropout 정규화와 Paper 2의 "학습률이 가장 중요한 하이퍼파라미터"라는 발견은, 내일 다룰 옵티마이저와 weight decay가 왜 모델 성능의 핵심 레버인지 이해하는 자연스러운 다리가 됩니다.
