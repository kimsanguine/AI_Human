# Foundational Papers — 일리야 서츠케버 추천 AI 핵심 논문 목록

> **Source:** [30 Papers — Ilya Sutskever's recommended reading list](https://30papers.com/) · [GeekNews 요약](https://news.hada.io/topic?id=31224)
> **Context:** 일리야 서츠케버(Ilya Sutskever)가 존 카맥(John Carmack)에게 추천했다고 알려진 현대 AI의 뼈대를 이루는 논문 모음. "이것들을 제대로 이해하면 중요한 것의 90%를 안다"고 언급된 리스트로, 컴퓨터 비전 → 순차 모델 → 어텐션/트랜스포머 → 메모리·관계·그래프 → 스케일링 → 정보이론/복잡성 → 생성 모델/AGI로 이어지는 사고의 흐름을 담는다.

## 이 폴더의 목적
이 커리큘럼(AI 엔지니어 과정)의 일일 논문 추천은 모듈·날짜 단위로 정리되어 있어, "리스트 전체를 하나의 독립된 읽기 경로"로 보기 어렵다. 이 폴더는 위 리스트에서 **아직 저장소에 없던 기초 논문들을 한곳에 모아 PDF로 보강**한 것이다. 이미 저장소 다른 위치에 있는 논문은 아래 표에 **기존 위치를 링크**로 표시했다.

> 참고: 리스트에는 논문 외에 강의 노트·블로그·튜토리얼(CS231n, Karpathy의 "The Unreasonable Effectiveness of RNNs", colah의 "Understanding LSTM Networks", "The Annotated Transformer", Kolmogorov Complexity 참고자료, Scott Aaronson의 "The First Law of Complexodynamics")도 포함되나, PDF 논문이 아니므로 이 폴더에는 담지 않았다.

---

## 1. 컴퓨터 비전과 합성곱 신경망

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 1 | ImageNet Classification with Deep CNNs (AlexNet) | Krizhevsky et al., 2012 | ✅ 저장소 보유 — [cnn-architectures 모듈](../2026-04-05-cnn-architectures-and-computer-vision/) (`imagenet-classification-krizhevsky-2012.pdf`) |
| 2 | Deep Residual Learning for Image Recognition (ResNet) | He et al., 2015 | ✅ 저장소 보유 — `deep-residual-learning-he-2015.pdf` (cnn-architectures) |
| 3 | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | He et al., 2016 | ⬇️ 추가됨 → `identity-mappings-resnet-he-2016.pdf` |
| 4 | [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) | Yu & Koltun, 2015 | ⬇️ 추가됨 → `dilated-convolutions-yu-koltun-2015.pdf` |

**소개:** CNN은 "공간을 읽는 법"을 배우는 구조다. AlexNet이 딥러닝 시대를 열었고, ResNet의 잔차 연결(residual connection)이 수백 층 규모의 학습을 가능케 했으며, Identity Mappings는 그 shortcut이 왜 잘 작동하는지 분석해 pre-activation 블록을 제안했다. Dilated Convolution은 해상도 손실 없이 수용 영역을 넓혀 dense prediction(분할 등)의 기반이 되었다.

---

## 2. 순차 모델과 장기 의존성

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 5 | [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329) | Zaremba et al., 2014 | ⬇️ 추가됨 → `rnn-regularization-zaremba-2014.pdf` |
| 6 | [Order Matters: Sequence to Sequence for Sets](https://arxiv.org/abs/1511.06391) | Vinyals et al., 2015 | ⬇️ 추가됨 → `order-matters-seq2seq-sets-vinyals-2015.pdf` |

**소개:** RNN/LSTM은 "시간을 기억하는 법"을 다룬다. RNN Regularization은 순환 연결이 아닌 비순환 연결에만 dropout을 적용해야 함을 밝혀 대형 순환망의 과적합을 줄였고, Order Matters는 본질적으로 순서가 없는 집합(set) 데이터를 순차 모델로 다룰 때의 문제를 분석했다.

---

## 3. 어텐션과 트랜스포머

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 7 | Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau) | Bahdanau et al., 2014 | ✅ 저장소 보유 — [attention 모듈](../../nlp-speech-data/2026-04-10-attention-mechanism-and-transformer/) (`neural-machine-translation-bahdanau-2014.pdf`) |
| 8 | Pointer Networks | Vinyals et al., 2015 | ✅ 저장소 보유 — `pointer-networks-vinyals-2015.pdf` (attention 모듈) |
| 9 | Attention Is All You Need (Transformer) | Vaswani et al., 2017 | ✅ 저장소 보유 — `attention-is-all-you-need-vaswani-2017.pdf` (attention 모듈) |

**소개:** 어텐션은 "필요한 정보를 골라 보는 법", 트랜스포머는 "그 과정을 병렬화하는 법"이다. Bahdanau 어텐션이 고정 요약 벡터의 병목을 풀었고, Pointer Networks는 출력이 입력의 위치를 가리키게 해 조합 최적화 문제에 길을 열었으며, Attention Is All You Need는 recurrence를 제거하고 self-attention만으로 현대 LLM의 구조적 기반을 세웠다.

---

## 4. 메모리, 관계 추론, 그래프

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 10 | Neural Turing Machines | Graves et al., 2014 | ✅ 저장소 보유 — [rnn-lstm 모듈](../2026-04-06-rnn-lstm-sequence-models/) (`neural-turing-machines-graves-2014.pdf`) |
| 11 | [A Simple Neural Network Module for Relational Reasoning](https://arxiv.org/abs/1706.01427) | Santoro et al., 2017 | ⬇️ 추가됨 → `relation-networks-relational-reasoning-santoro-2017.pdf` |
| 12 | [Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822) | Santoro et al., 2018 | ⬇️ 추가됨 → `relational-rnn-santoro-2018.pdf` |
| 13 | [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) | Gilmer et al., 2017 | ⬇️ 추가됨 → `neural-message-passing-gilmer-2017.pdf` |

**소개:** 이 계열은 "객체들의 관계를 계산하는 법"을 다룬다. Neural Turing Machine은 미분 가능한 외부 메모리를, Relation Network는 객체 쌍 관계 추론 모듈을, Relational RNN은 self-attention 기반 상호작용 메모리를 제안했다. Neural Message Passing은 여러 그래프 신경망을 message passing 프레임워크로 통합해 GNN 연구의 기틀을 정리했다.

---

## 5. 대규모 학습과 스케일링 법칙

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 14 | Scaling Laws for Neural Language Models | Kaplan et al., 2020 | ✅ 저장소 보유 — [llm-nlg 모듈](../../llm-nlg/2026-04-16-gpt-architecture-scaling-laws/) (`scaling-laws-neural-language-models-kaplan-2020.pdf`) |
| 15 | [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965) | Huang et al., 2018 | ⬇️ 추가됨 → `gpipe-huang-2018.pdf` |
| 16 | Deep Speech 2 | Amodei et al., 2015 | ✅ 저장소 보유 — [speech 모듈](../../nlp-speech-data/2026-04-12-speech-recognition-fundamentals/) (`deep-speech-2-amodei-2015.pdf`) |

**소개:** "크게 훈련하는 법"에 대한 논문들. Scaling Laws는 손실이 모델·데이터·연산량에 따라 power law로 감소함을 실증해 대형 모델 투자의 근거가 되었고, GPipe는 pipeline parallelism으로 거대 모델을 여러 장치에 나눠 학습하는 실용적 방법을, Deep Speech 2는 end-to-end 음성인식의 확장 가능성을 보였다.

---

## 6. 정보이론, 압축, 복잡성

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 17 | [Keeping Neural Networks Simple by Minimizing the Description Length of the Weights](https://www.cs.toronto.edu/~hinton/absps/colt93.pdf) | Hinton & van Camp, 1993 | ⬇️ 추가됨 → `keeping-nn-simple-mdl-hinton-1993.pdf` |
| 18 | [A Tutorial Introduction to the Minimum Description Length Principle](https://arxiv.org/abs/math/0406077) | Grünwald, 2004 | ⬇️ 추가됨 → `mdl-tutorial-grunwald-2004.pdf` |
| 19 | [Quantifying the Rise and Fall of Complexity in Closed Systems: The Coffee Automaton](https://arxiv.org/abs/1405.6903) | Aaronson et al., 2014 | ⬇️ 추가됨 → `coffee-automaton-complexity-aaronson-2014.pdf` |

**소개:** "왜 학습이 압축과 일반화의 문제인지 생각하는 법". 좋은 모델은 데이터를 짧게 설명(적은 비트)하는 가중치를 갖는다는 MDL 관점이 핵심이다. Hinton & van Camp가 신경망 일반화를 설명 길이와 연결했고, Grünwald의 튜토리얼이 MDL 원리를 정리했으며, Coffee Automaton은 닫힌 계에서 복잡성이 증가했다 감소하는 현상을 정량화한다.

---

## 7. 생성 모델과 보편적 지능

| # | 논문 | 저자·연도 | 상태 |
|---|------|-----------|------|
| 20 | [Variational Lossy Autoencoder](https://arxiv.org/abs/1611.02731) | Chen et al., 2016 | ⬇️ 추가됨 → `variational-lossy-autoencoder-chen-2016.pdf` |
| 21 | [Machine Super Intelligence](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf) | Legg, 2008 (PhD thesis) | ⬇️ 추가됨 → `machine-super-intelligence-legg-2008.pdf` |

**소개:** VAE에 autoregressive decoder를 결합한 Variational Lossy Autoencoder는 latent code가 어떤 정보를 보존할지 제어하는 방법을 제시해 표현 학습과 정보 보존의 균형을 다룬다. Machine Super Intelligence(Legg의 박사논문)는 기계 지능의 보편적 측정과 강력한 에이전트의 성질을 이론적으로 탐구한, AGI 논의의 형식적 기반 중 하나다.

---

## 요약: 리스트 커버리지
- **리스트의 PDF 논문 21편 중** — 기존 저장소 보유 8편 + 이 폴더에 신규 추가 13편 = **21편 전부 확보**
- **제외된 6개 항목** — 강의/블로그/튜토리얼/참고자료(CS231n, Unreasonable Effectiveness of RNNs, Understanding LSTM Networks, The Annotated Transformer, Kolmogorov Complexity, The First Law of Complexodynamics)는 PDF 논문이 아니어서 미포함
