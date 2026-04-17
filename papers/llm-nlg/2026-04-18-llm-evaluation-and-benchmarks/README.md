# Daily AI Paper Recommendations

> **Date:** 2026-04-18
> **Module:** LLM for Natural Language Generation
> **Topic:** LLM Evaluation and Benchmarks

---

## Paper 1 (Classic): Measuring Massive Multitask Language Understanding (MMLU)
- **Authors:** Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, Jacob Steinhardt
- **Year:** 2020
- **arXiv:** https://arxiv.org/abs/2009.03300
- **PDF:** [./mmlu-hendrycks-2020.pdf](./mmlu-hendrycks-2020.pdf)
- **Citation Count:** ~6,000+

### 요약
MMLU는 대규모 언어 모델의 지식과 추론 능력을 57개 과목에 걸쳐 측정하는 벤치마크이다. STEM, 인문학, 사회과학, 전문 분야 등 다양한 도메인의 객관식 문제로 구성되어 있으며, 모델의 제로샷 및 퓨샷 성능을 평가한다. 이 벤치마크는 LLM 평가의 사실상 표준(de facto standard)으로 자리잡았다.

### 핵심 기여
- 57개 학문 분야를 아우르는 대규모 다중 작업(multitask) 평가 프레임워크를 최초로 제안
- 언어 모델이 세계 지식(world knowledge)을 얼마나 학습했는지 정량적으로 측정하는 방법론 확립
- 모델 크기와 사전 학습 데이터 양에 따른 성능 스케일링 트렌드를 체계적으로 분석

### 이 논문이 중요한 이유
MMLU는 GPT-4, Claude, Gemini 등 모든 주요 LLM의 성능을 비교하는 핵심 벤치마크로, AI 엔지니어라면 반드시 이해해야 하는 평가 체계이다. 모델의 "일반 지능"을 측정하는 첫 번째 시도로서, 이후 등장한 거의 모든 LLM 벤치마크의 기반이 되었다. 모델 선택, 파인튜닝 전략 수립, 성능 회귀 테스트 등 실무에서 직접적으로 활용된다.

### 사전 지식
- 언어 모델의 기본 개념 (GPT 아키텍처, 사전 학습/파인튜닝)
- 퓨샷 러닝(few-shot learning)과 인컨텍스트 러닝의 원리
- 분류 문제의 평가 메트릭 (accuracy, F1 등)

### 관련 논문
- [Language Models are Few-Shot Learners / GPT-3 (Brown et al., 2020)](https://arxiv.org/abs/2005.14165)
- [HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019)](https://arxiv.org/abs/1905.07830)
- [ARC: AI2 Reasoning Challenge (Clark et al., 2018)](https://arxiv.org/abs/1803.05457)

### 실무 적용
LLM을 프로덕션에 도입할 때 모델 간 비교 평가의 기준으로 활용된다. 특히 도메인 특화 모델을 개발할 때 MMLU의 특정 카테고리 점수를 추적하여 모델의 지식 수준을 모니터링할 수 있다. 또한 파인튜닝 후 일반 지식이 손실되지 않았는지 확인하는 "catastrophic forgetting" 검증에도 널리 사용된다.

---

## Paper 2 (Classic): Evaluating Large Language Models Trained on Code (HumanEval / Codex)
- **Authors:** Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, Wojciech Zaremba
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2107.03374
- **PDF:** [./humaneval-chen-2021.pdf](./humaneval-chen-2021.pdf)
- **Citation Count:** ~5,000+

### 요약
이 논문은 GitHub의 공개 코드로 파인튜닝한 GPT 기반 언어 모델인 Codex를 소개하고, 코드 생성 능력을 평가하기 위한 HumanEval 벤치마크를 제안한다. HumanEval은 164개의 Python 프로그래밍 문제로 구성되어 있으며, docstring으로부터 함수를 합성하는 능력을 유닛 테스트를 통해 "기능적 정확성(functional correctness)" 관점에서 평가한다.

### 핵심 기여
- 코드 생성의 기능적 정확성을 측정하는 최초의 체계적 벤치마크(HumanEval) 제안
- pass@k 메트릭 도입: 반복 샘플링을 통해 최소 하나의 정확한 솔루션을 생성할 확률을 측정
- Codex 모델의 코드 생성 능력이 모델 크기에 따라 로그 선형적으로 향상됨을 입증

### 이 논문이 중요한 이유
HumanEval은 LLM의 코드 생성 능력을 평가하는 가장 대표적인 벤치마크로, GitHub Copilot의 기반이 된 Codex 모델과 함께 발표되었다. AI 코딩 어시스턴트의 성능 비교에 필수적이며, "텍스트 매칭"이 아닌 "실행 기반 평가"라는 패러다임을 확립했다. 이는 현재 SWE-bench, LiveCodeBench 등 모든 코드 평가 벤치마크의 방법론적 근간이 된다.

### 사전 지식
- GPT 아키텍처와 언어 모델의 기본 이해
- Python 프로그래밍 기본 (함수, docstring, 유닛 테스트)
- 확률적 샘플링과 temperature의 개념

### 관련 논문
- [Competition-Level Code Generation with AlphaCode (Li et al., 2022)](https://arxiv.org/abs/2203.07814)
- [CodeBERTa: A Pre-Trained Model for Programming Languages (Feng et al., 2020)](https://arxiv.org/abs/2002.08155)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (Jimenez et al., 2023)](https://arxiv.org/abs/2310.06770)

### 실무 적용
AI 코딩 도구(Copilot, Cursor, Cody 등)의 성능을 비교하고 평가하는 데 핵심적으로 사용된다. 사내 코드 생성 모델을 개발하거나 파인튜닝할 때, HumanEval 또는 이를 확장한 벤치마크(HumanEval+, MultiPL-E)를 활용하여 성능을 측정한다. pass@k 메트릭은 실제 사용자 경험(여러 번 시도하면 원하는 코드를 얻을 확률)을 잘 반영하여 프로덕트 의사결정에 유용하다.

---

## Paper 3 (Recent): MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
- **Authors:** Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Beutel, Xianjun Yang, Benyou Wang, Murong Yue, Bohao Peng, Zheng Feng, Ming Fan, Wenhu Chen
- **Year:** 2024
- **arXiv:** https://arxiv.org/abs/2406.01574
- **PDF:** [./mmlu-pro-wang-2024.pdf](./mmlu-pro-wang-2024.pdf)
- **Citation Count:** ~200+ (NeurIPS 2024)

### 요약
MMLU-Pro는 기존 MMLU 벤치마크의 한계를 극복하기 위해 설계된 강화된 평가 프레임워크이다. 선택지를 4개에서 10개로 확대하고, 추론이 필요한 고난이도 문제를 통합하여 14개 학문 분야에서 12,000개 이상의 문제를 포함한다. 기존 MMLU 대비 정확도가 16~33% 하락하며, 프롬프트 변화에 대한 안정성이 크게 향상되었다.

### 핵심 기여
- 기존 MMLU의 "노이즈 포함" 및 "지식 편향" 문제를 해결하여 더 신뢰성 있는 평가 제공
- 10지선다 방식 도입으로 무작위 추측(random guessing)의 영향을 10%로 줄여 변별력 향상
- Chain-of-Thought(CoT) 프롬프팅 시 성능 향상이 더 뚜렷하게 나타나, 추론 능력 평가에 적합함을 입증

### 이 논문이 중요한 이유
최신 LLM들이 MMLU에서 90%+ 정확도를 달성하면서 벤치마크의 변별력이 떨어진 상황에서, MMLU-Pro는 차세대 표준 벤치마크로 빠르게 채택되고 있다. NeurIPS 2024에 발표되었으며, 프롬프트 민감도 문제를 해결하여 더 공정한 모델 비교를 가능하게 한다. AI 엔지니어로서 모델 평가 파이프라인을 설계할 때 반드시 고려해야 할 최신 벤치마크이다.

### 사전 지식
- MMLU 벤치마크의 구조와 평가 방법론 (Paper 1 참조)
- Chain-of-Thought (CoT) 프롬프팅의 개념
- LLM의 인컨텍스트 러닝과 퓨샷 프롬프팅 이해

### 관련 논문
- [Measuring Massive Multitask Language Understanding / MMLU (Hendrycks et al., 2020)](https://arxiv.org/abs/2009.03300)
- [MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs (2024)](https://arxiv.org/abs/2409.02257)
- [LiveBench: A Challenging, Contamination-Free LLM Benchmark (White et al., 2024)](https://arxiv.org/abs/2406.19314)

### 실무 적용
LLM을 선정하거나 비교 평가할 때, MMLU만으로는 최신 모델 간 차이를 변별하기 어려운 상황에서 MMLU-Pro를 보조 벤치마크로 활용할 수 있다. 특히 추론 능력이 중요한 B2B SaaS 제품(법률, 의료, 금융 도메인)에서 모델 선택 시 MMLU-Pro의 도메인별 점수가 더 유의미한 지표가 된다. 또한 프롬프트 엔지니어링의 효과를 측정할 때 안정적인 기준선을 제공한다.

---

## 추천 읽기 순서

1. **MMLU (Hendrycks et al., 2020)** — LLM 평가의 기초를 다지는 논문. 57개 과목에 걸친 다중 작업 평가라는 개념을 먼저 이해하면, 이후 논문들의 맥락이 명확해진다.
2. **HumanEval / Codex (Chen et al., 2021)** — 텍스트 생성을 넘어 코드 생성이라는 새로운 평가 차원을 이해한다. "실행 기반 평가"와 pass@k 메트릭은 LLM 평가 방법론의 중요한 확장이다.
3. **MMLU-Pro (Wang et al., 2024)** — MMLU의 한계를 파악하고, 차세대 벤치마크가 어떤 방향으로 진화하고 있는지 이해한다. 최신 모델 평가에 직접 활용 가능한 실무적 가치가 높다.

## 핵심 테이크어웨이

- **평가는 진화한다**: MMLU → MMLU-Pro의 진화는, 모델이 발전하면 벤치마크도 함께 업그레이드되어야 함을 보여준다. "벤치마크 포화(saturation)" 문제는 AI 엔지니어가 항상 인식해야 할 과제이다.
- **다차원 평가가 필수**: 지식(MMLU), 코드(HumanEval), 추론(MMLU-Pro) 등 다양한 차원에서 모델을 평가해야 실제 성능을 파악할 수 있다. 단일 벤치마크에 의존하는 것은 위험하다.
- **실행 기반 평가의 중요성**: HumanEval이 도입한 "코드를 실제로 실행하여 검증"하는 패러다임은, 단순 텍스트 매칭을 넘어선 신뢰성 높은 평가 방법론의 시작점이다.
- **프롬프트 민감도 인식**: MMLU-Pro가 해결하려는 핵심 문제 중 하나는 프롬프트 변화에 따른 점수 변동이다. 실무에서 모델을 비교할 때 이런 변수를 통제하는 것이 중요하다.

## 다음 토픽과의 연결

다음 토픽인 **Efficient LLM: Quantization and Distillation**에서는 대규모 LLM을 효율적으로 운용하기 위한 경량화 기법을 다룬다. 오늘 학습한 벤치마크들은 경량화된 모델이 원본 대비 얼마나 성능을 유지하는지 측정하는 핵심 도구가 된다. LoRA, QLoRA 등으로 파인튜닝/양자화한 모델의 품질을 MMLU, HumanEval, MMLU-Pro로 검증하는 것이 표준 워크플로우이다.
