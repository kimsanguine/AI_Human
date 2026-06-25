# Daily AI Paper Recommendations

> **Date:** 2026-06-26
> **Module:** Module 3: Machine Learning and Deep Learning
> **Topic:** RNN, LSTM and Sequence Models

---

## Paper 1 (Classic): Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **Authors:** Junyoung Chung, Çağlar Gülçehre, Kyunghyun Cho, Yoshua Bengio
- **Year:** 2014
- **arXiv:** https://arxiv.org/abs/1412.3555
- **PDF:** [./gated-rnn-evaluation-chung-2014.pdf](./gated-rnn-evaluation-chung-2014.pdf)
- **Citation Count:** approx. 14,000+

### 요약
LSTM과 GRU처럼 게이팅 메커니즘을 가진 순환 유닛이 전통적인 tanh 유닛에 비해 시퀀스 모델링에서 얼마나 우수한지를 실증적으로 비교한 논문이다. 폴리포닉 음악 모델링과 음성 신호 모델링 과제에서 세 유닛을 동일 조건으로 평가했다. 결론적으로 게이트 기반 유닛이 tanh보다 확실히 우월하며, GRU는 LSTM보다 파라미터가 적으면서도 비슷한 성능을 낸다는 것을 보였다.

### 핵심 기여
- LSTM, GRU, tanh RNN을 동일한 실험 환경에서 직접 비교하여 게이트 유닛의 우월성을 실증함
- GRU라는 비교적 단순한 게이트 구조가 LSTM에 근접한 성능을 낸다는 점을 널리 알림
- 게이팅이 장기 의존성 학습과 기울기 흐름 안정화에 기여함을 과제별로 확인함

### 이 논문이 중요한 이유
LSTM과 GRU 중 무엇을 쓸지는 실무에서 늘 반복되는 선택이다. 이 논문은 그 의사결정의 실증적 근거를 제공한 최초의 표준 레퍼런스로, "GRU vs LSTM" 논의의 출발점이 되었다. 아키텍처 복잡도와 성능 사이의 트레이드오프를 데이터로 사고하는 방법을 보여준다.

### 사전 지식
- 기본 RNN 구조와 BPTT(Backpropagation Through Time)
- 기울기 소실/폭발 문제와 게이팅의 역할
- LSTM의 cell state, GRU의 update/reset gate 개념

### 관련 논문
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Learning Phrase Representations using RNN Encoder–Decoder (Cho et al., 2014)](https://arxiv.org/abs/1406.1078)

### 실무 적용
시계열 예측, 음성 인식, 텍스트 분류 등에서 LSTM 대신 GRU를 선택해 연산량과 메모리를 줄이는 결정의 근거가 된다. 리소스가 제한된 엣지/모바일 환경이나 학습 속도가 중요한 프로토타이핑 단계에서 GRU를 우선 검토하는 관행이 여기서 비롯됐다.

---

## Paper 2 (Classic): Pointer Networks
- **Authors:** Oriol Vinyals, Meire Fortunato, Navdeep Jaitly
- **Year:** 2015
- **arXiv:** https://arxiv.org/abs/1506.03134
- **PDF:** [./pointer-networks-vinyals-2015.pdf](./pointer-networks-vinyals-2015.pdf)
- **Citation Count:** approx. 6,000+

### 요약
출력 토큰이 입력 시퀀스의 "위치"를 가리키는 새로운 시퀀스-투-시퀀스 구조를 제안한다. 기존 seq2seq는 출력 어휘 크기가 고정되어야 하지만, 정렬·볼록 껍질·외판원 문제처럼 출력 클래스 수가 입력 길이에 따라 달라지는 문제는 다룰 수 없었다. Pointer Network는 어텐션을 분포가 아닌 "포인터"로 사용해 입력 원소를 직접 선택함으로써 이 한계를 해결한다.

### 핵심 기여
- 어텐션을 가중합이 아니라 입력 위치를 선택하는 포인터로 재해석함
- 가변 길이 출력 사전(output dictionary) 문제를 우아하게 해결함
- 조합 최적화(정렬, convex hull, TSP) 문제를 신경망으로 근사하는 길을 염

### 이 논문이 중요한 이유
포인터 메커니즘은 이후 추상 요약의 copy mechanism, CopyNet, 그리고 현대 LLM의 retrieval·tool-use에서 입력 토큰을 직접 참조하는 아이디어로 계승됐다. 어텐션의 활용 범위를 "정렬"에서 "선택"으로 확장한 전환점이다.

### 사전 지식
- seq2seq 인코더-디코더 구조
- Bahdanau/Luong 어텐션의 기본 동작
- 소프트맥스 기반 확률 분포와 argmax 선택의 차이

### 관련 논문
- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [Get To The Point: Summarization with Pointer-Generator Networks (See et al., 2017)](https://arxiv.org/abs/1704.04368)

### 실무 적용
추출형/하이브리드 텍스트 요약, 개체명·슬롯 추출, 입력에 없는 OOV 단어를 그대로 복사해야 하는 대화/QA 시스템, 경로 최적화·스케줄링 같은 조합 최적화 휴리스틱 학습에 활용된다. "입력의 일부를 그대로 가리켜 출력"하는 모든 태스크의 기본 패턴이다.

---

## Paper 3 (Recent): Titans: Learning to Memorize at Test Time
- **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni
- **Year:** 2025
- **arXiv:** https://arxiv.org/abs/2501.00663
- **PDF:** [./titans-memorize-test-time-behrouz-2025.pdf](./titans-memorize-test-time-behrouz-2025.pdf)
- **Citation Count:** approx. 200+ (2025년 발표, 빠르게 증가 중)

### 요약
Transformer의 고정 컨텍스트 한계와 선형 순환 모델의 메모리 한계를 동시에 넘으려는 새로운 하이브리드 아키텍처 Titans를 제안한다. 핵심은 추론(test time) 중에도 자기 가중치를 갱신하며 기억하는 Neural Long-Term Memory Module(LMM)이다. 입력의 "놀라움(surprise)"을 기울기 신호로 측정하고 모멘텀과 적응적 망각을 결합해 무엇을 기억하고 버릴지 학습한다.

### 핵심 기여
- 추론 시점에 가중치를 업데이트하며 장기 기억을 형성하는 메타-인컨텍스트 학습 모듈 제안
- surprise 기반 메모리 갱신과 적응적 forgetting으로 메모리 오버플로 방지
- 어텐션을 단기 기억, LMM을 장기 기억으로 결합해 200만+ 토큰의 초장기 컨텍스트로 확장

### 이 논문이 중요한 이유
"컨텍스트 윈도우를 늘린다"는 접근을 넘어 "모델이 추론 중 스스로 기억을 학습한다"는 패러다임 전환을 보여준다. Mamba/xLSTM 같은 순환 계열의 흐름을 잇되, test-time learning과 결합한 점에서 차세대 장기 메모리 에이전트 설계에 직접 영향을 준다.

### 사전 지식
- Transformer 어텐션과 컨텍스트 길이의 계산 복잡도
- State Space Model(Mamba)과 선형 순환 모델의 기본 개념
- test-time training / meta-learning과 기울기 기반 적응의 직관

### 관련 논문
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752)
- [xLSTM: Extended Long Short-Term Memory (Beck et al., 2024)](https://arxiv.org/abs/2405.04517)

### 실무 적용
초장기 문서/로그 분석, 지속적으로 사용자를 기억해야 하는 개인화 에이전트, 긴 코드베이스나 대화 히스토리를 다루는 RAG 대체·보완 메모리 계층 설계에 응용된다. 고정 컨텍스트 비용을 줄이면서 장기 상태를 유지해야 하는 Agentic AI 제품의 메모리 아키텍처 후보로 검토할 만하다.

---

## 추천 읽기 순서
1. **Empirical Evaluation of Gated RNNs (Chung, 2014)** — 순환 유닛의 기본기와 LSTM/GRU 선택 감각을 먼저 잡는다.
2. **Pointer Networks (Vinyals, 2015)** — 어텐션을 "선택"으로 확장하며 seq2seq의 한계와 돌파를 이해한다.
3. **Titans (Behrouz, 2025)** — 순환·어텐션·test-time 메모리가 결합된 최신 흐름으로 마무리한다.

## 핵심 테이크어웨이
- 게이팅은 순환 모델의 장기 의존성 학습을 가능케 한 핵심 장치이며, GRU는 단순함과 성능의 좋은 균형점이다.
- 어텐션은 정렬을 넘어 입력을 직접 가리키는 "포인터"로 일반화될 수 있고, 이는 copy·retrieval의 뿌리가 된다.
- 2025년의 화두는 컨텍스트 확장에서 "추론 중 학습하는 메모리"로 이동하고 있으며, 이는 에이전트 설계의 핵심 축이다.

## 다음 토픽과의 연결
다음 토픽인 **Optimization and Regularization(최적화와 정규화)**는 오늘 다룬 순환·메모리 모델을 실제로 "안정적으로 학습"시키는 도구를 제공한다. Titans의 test-time 가중치 갱신, GRU의 기울기 안정성 모두 옵티마이저와 정규화 전략과 직결되므로, 학습 동역학 관점에서 자연스럽게 이어진다.
