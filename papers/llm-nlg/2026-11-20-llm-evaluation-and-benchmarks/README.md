# Daily AI Paper Recommendations

> **Date:** 2026-11-20
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** LLM Evaluation and Benchmarks

> 이번 사이클에서는 이전에 다룬 벤치마크(MMLU, HumanEval, HELM, MT-Bench, GLUE, BIG-bench, HellaSwag, GSM8K, MATH, ARC, TruthfulQA, G-Eval, IFEval, τ-bench 등)와 겹치지 않도록 **"실제 업무를 끝까지 해내는가(SWE-bench)" → "전문가도 검색으로 못 푸는 문제를 푸는가(GPQA)" → "LLM 심판의 길이 편향을 어떻게 걷어내는가(Length-Controlled AlpacaEval)"** 라는 흐름으로 골랐습니다. 질문은 하나입니다: **"벤치마크 점수가 올랐을 때, 실제로 무엇이 좋아진 것인가?"**

---

## Paper 1 (Classic): SWE-bench: Can Language Models Resolve Real-World GitHub Issues?
- **Authors:** Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, Karthik Narasimhan
- **Year:** 2023 (ICLR 2024 Oral)
- **arXiv:** https://arxiv.org/abs/2310.06770
- **PDF:** [./swe-bench-jimenez-2023.pdf](./swe-bench-jimenez-2023.pdf)
- **Citation Count:** ~1,500+

### 요약
HumanEval처럼 함수 하나를 짜는 문제는 실제 개발 업무와 거리가 멉니다. SWE-bench는 12개 인기 Python 저장소(Django, scikit-learn, sympy 등)의 실제 GitHub 이슈와 그 이슈를 해결한 PR 2,294개를 모아, 모델이 저장소 전체를 보고 패치를 만들면 실제 단위 테스트(fail→pass)로 해결 여부를 판정합니다. 공개 당시 최고 모델(Claude 2)의 해결률은 약 2%에 불과했습니다.

### 핵심 기여
- **실행 기반 검증:** 정답 코드와의 문자열 비교가 아니라, 이슈를 재현하는 테스트가 통과하는지(FAIL_TO_PASS)와 기존 테스트가 깨지지 않는지(PASS_TO_PASS)로 평가
- **저장소 규모 컨텍스트:** 평균 수천 개 파일·수십만 줄 코드베이스에서 관련 파일을 찾고 여러 파일을 동시에 수정해야 하는 현실적 과제
- **지속 갱신 가능한 수집 파이프라인:** 새로운 이슈·PR로 벤치마크를 계속 만들 수 있어 데이터 오염(contamination)에 상대적으로 강함
- 학습용 데이터셋 SWE-bench-train과 파인튜닝 모델 SWE-Llama 공개

### 이 논문이 중요한 이유
Claude Code, Codex, Cursor 등 코딩 에이전트의 성능을 비교할 때 사실상 표준이 된 지표(SWE-bench Verified)의 출발점입니다. "모델이 코드를 잘 짠다"는 막연한 주장을 "실제 이슈의 몇 %를 테스트 통과 수준으로 해결하는가"라는 측정 가능한 질문으로 바꿨습니다. 에이전트를 만드는 AI 엔지니어라면 평가 설계의 모범 사례로 반드시 봐야 합니다.

### 사전 지식
- HumanEval과 pass@k 지표 (이전 사이클 Day 16)
- Git diff/patch, 단위 테스트(pytest) 기본 개념
- BM25 등 검색 기반 컨텍스트 선택(RAG 기초)

### 관련 논문
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering (Yang et al., 2024)](https://arxiv.org/abs/2405.15793)
- [Evaluating Large Language Models Trained on Code / HumanEval (Chen et al., 2021)](https://arxiv.org/abs/2107.03374)
- [SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains? (Yang et al., 2024)](https://arxiv.org/abs/2410.03859)
- [Agentless: Demystifying LLM-based Software Engineering Agents (Xia et al., 2024)](https://arxiv.org/abs/2407.01489)

### 실무 적용
자사 저장소에서 이미 해결된 이슈·PR 50~100개를 골라 "미니 SWE-bench"를 만들면, 코딩 에이전트 도입·모델 교체·프롬프트 변경의 효과를 감이 아니라 해결률로 비교할 수 있습니다. 핵심은 정답 코드와의 유사도가 아니라 **테스트 통과 여부**로 판정하는 것, 그리고 원래 테스트가 깨지지 않는지(회귀)를 함께 보는 것입니다.

---

## Paper 2 (Classic): GPQA: A Graduate-Level Google-Proof Q&A Benchmark
- **Authors:** David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
- **Year:** 2023 (COLM 2024)
- **arXiv:** https://arxiv.org/abs/2311.12022
- **PDF:** [./gpqa-rein-2023.pdf](./gpqa-rein-2023.pdf)
- **Citation Count:** ~1,200+

### 요약
MMLU가 포화되면서 "모델이 사람 전문가 수준을 넘었는가"를 측정할 더 어려운 문제가 필요해졌습니다. GPQA는 생물·물리·화학 박사급 전문가가 작성한 448개의 객관식 문제로, 해당 분야 전문가는 약 65%(명백한 실수 제외 시 74%)를 맞히지만, 다른 분야 박사가 웹 검색을 30분 이상 자유롭게 써도 34% 정도밖에 못 맞힙니다. 공개 당시 GPT-4는 약 39%였습니다.

### 핵심 기여
- **Google-proof 설계:** 비전문가가 검색으로 답을 찾을 수 없도록 문제를 만들고 검증하는 다단계 파이프라인(작성 → 전문가 검증 → 수정 → 2차 검증 → 비전문가 검증)
- **문제 품질의 정량화:** 전문가 정답률과 비전문가 정답률의 차이로 "진짜 전문 지식이 필요한 문제"인지 측정
- **Diamond 서브셋(198문항):** 전문가는 맞히고 비전문가는 틀린 고품질 문항만 모아, 이후 o1·Claude·Gemini 등 추론 모델의 대표 지표가 됨
- **Scalable Oversight 연구 동기:** "사람보다 잘하는 AI를 사람이 어떻게 감독할 것인가"라는 문제를 실험할 수 있는 테스트베드 제시

### 이 논문이 중요한 이유
오늘날 모델 출시 발표에서 빠지지 않는 "GPQA Diamond" 점수의 원천입니다. 동시에 벤치마크를 **어떻게 만들어야 신뢰할 수 있는지**(작성자·검증자 분리, 비전문가 대조군, 오답 선택지 품질 관리)를 보여주는 데이터 구축 교과서이기도 합니다. 점수만 보지 말고 "이 점수가 무엇을 의미하는가"를 해석하는 눈을 길러줍니다.

### 사전 지식
- MMLU 등 객관식 벤치마크와 포화(saturation) 문제
- Few-shot / Chain-of-Thought 프롬프팅 (Day 18)
- 평가자 간 일치도(inter-annotator agreement) 개념

### 관련 논문
- [Measuring Massive Multitask Language Understanding / MMLU (Hendrycks et al., 2020)](https://arxiv.org/abs/2009.03300)
- [Humanity's Last Exam (Phan et al., 2025)](https://arxiv.org/abs/2501.14249)
- [Measuring Progress on Scalable Oversight for Large Language Models (Bowman et al., 2022)](https://arxiv.org/abs/2211.03540)
- [Debate Helps Supervise Unreliable Experts (Michael et al., 2023)](https://arxiv.org/abs/2311.08702)

### 실무 적용
의료·법률·금융처럼 전문 도메인 AI를 만든다면, 사내 전문가에게 "검색으로는 답이 안 나오는" 문제를 쓰게 하고 비전문가 대조군으로 난이도를 검증하는 GPQA 방식이 그대로 쓸 수 있는 평가셋 구축법입니다. 문항 수가 적어도(수백 개) 품질이 높으면 모델 선택 의사결정에 충분합니다. 단, 문항이 적을수록 신뢰구간이 넓으니 1~2%p 차이로 결론 내리지 마세요.

---

## Paper 3 (Recent): Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators
- **Authors:** Yann Dubois, Balázs Galambosi, Percy Liang, Tatsunori B. Hashimoto
- **Year:** 2024 (COLM 2024)
- **arXiv:** https://arxiv.org/abs/2404.04475
- **PDF:** [./length-controlled-alpacaeval-dubois-2024.pdf](./length-controlled-alpacaeval-dubois-2024.pdf)
- **Citation Count:** ~600+

### 요약
GPT-4 같은 LLM 심판은 내용이 같아도 **더 긴 답변을 선호**하는 편향이 있어, 모델이 답변을 길게만 만들어도 AlpacaEval 승률이 올라가는 "게이밍"이 가능했습니다. 이 논문은 회귀(GLM) 모델로 승률을 예측할 때 길이 차이 항을 분리한 뒤, "길이가 같았다면 승률이 얼마였을까"라는 반사실적(counterfactual) 질문에 답하는 방식으로 길이 편향을 제거합니다. 그 결과 Chatbot Arena와의 Spearman 상관이 0.94에서 0.98로 올라갔습니다.

### 핵심 기여
- **단순하고 해석 가능한 보정:** 모델 실력·지시문 난이도·길이 차이 세 항으로 심판의 선호를 모델링하고, 길이 항만 0으로 두어 보정 승률(LC win rate) 계산
- **게이밍 저항성:** "간결하게 답해" / "자세히 답해" 프롬프트만으로 승률이 크게 흔들리던 문제를 크게 완화
- **인간 선호와의 정합성 향상:** Chatbot Arena 순위와의 상관을 높여, 저비용(수 달러·수 분) 자동 평가의 신뢰도를 개선
- 이 방식이 AlpacaEval 2.0 공식 지표로 채택됨

### 이 논문이 중요한 이유
LLM-as-a-Judge는 이제 RAG·챗봇·에이전트 평가의 기본 도구가 됐지만, 심판 자체의 편향을 다루지 않으면 "점수는 올랐는데 사용자 만족은 그대로"인 상황이 생깁니다. 이 논문은 편향을 **통계적으로 모델링하고 제거하는** 실용적인 방법을 보여주며, 평가 지표 자체를 의심하고 검증하는 태도를 가르쳐줍니다.

### 사전 지식
- LLM-as-a-Judge, MT-Bench/Chatbot Arena (이전 사이클 Day 16)
- 로지스틱 회귀와 반사실적(counterfactual) 추론의 기초
- 순위 상관(Spearman) 개념

### 관련 논문
- [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback (Dubois et al., 2023)](https://arxiv.org/abs/2305.14387)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena (Zheng et al., 2023)](https://arxiv.org/abs/2306.05685)
- [From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder (Li et al., 2024)](https://arxiv.org/abs/2406.11939)
- [A Long Way to Go: Investigating Length Correlations in RLHF (Singhal et al., 2023)](https://arxiv.org/abs/2310.03716)

### 실무 적용
사내 LLM 심판으로 A/B 테스트를 할 때, 답변 길이(토큰 수)를 함께 기록하고 승률을 길이 차이로 회귀해 보세요. 길이 계수가 크다면 심판이 "긴 답"을 "좋은 답"으로 착각하고 있다는 신호입니다. 같은 원리로 위치 편향(먼저 나온 답 선호), 포맷 편향(마크다운·불릿 선호)도 공변량으로 넣어 점검할 수 있습니다. 특히 RLHF/DPO로 튜닝한 모델은 답변이 길어지는 경향이 있으니 보정 지표를 기본으로 두는 것이 안전합니다.

---

## 추천 읽기 순서
1. **GPQA** — 벤치마크를 "어떻게 만들어야 믿을 수 있는가"를 가장 명확하게 보여줌. 데이터 구축 파이프라인에 집중해서 읽기
2. **SWE-bench** — 정답 비교가 아닌 실행 결과로 평가하는 방식, 에이전트 시대의 평가 설계 이해
3. **Length-Controlled AlpacaEval** — 자동 평가자(LLM 심판)의 편향을 어떻게 측정하고 제거하는지. 짧고 실무 적용이 바로 가능

## 핵심 테이크어웨이
- **Q. 좋은 벤치마크의 조건은?** 정답을 검증할 수 있어야 하고(SWE-bench의 테스트), 쉽게 찾아볼 수 없어야 하며(GPQA의 Google-proof), 측정 도구 자체의 편향이 통제되어야 한다(LC AlpacaEval).
- **Q. 점수가 올랐다면 무엇이 좋아진 것인가?** 실력이 늘었는지, 길이·포맷 같은 표면적 특징이 심판을 속였는지, 데이터 오염인지를 먼저 의심하라.
- **Q. 공개 벤치마크를 그대로 믿어도 되나?** 출발점으로 삼되, 자사 이슈·전문가 문항·실사용 대화로 만든 내부 평가셋이 실제 의사결정 기준이 되어야 한다.
- **Q. 평가 비용은 어떻게 줄이나?** LLM 심판으로 비용을 줄이되, 사람 판단(Arena)과의 상관을 주기적으로 확인하고 알려진 편향을 통계적으로 보정하라.

## 다음 토픽과의 연결
다음 토픽 **Efficient LLM: Quantization and Distillation**에서는 LoRA, QLoRA, GPTQ 등으로 모델을 작고 빠르게 만드는 방법을 다룹니다. 경량화된 모델이 "얼마나 성능을 잃었는가"는 MMLU 같은 지식 벤치마크보다 SWE-bench처럼 여러 단계를 거치는 실행형 과제나 GPQA 같은 고난도 추론에서 먼저 드러나는 경우가 많습니다. 또한 증류(distillation) 과정에서 교사 모델의 장황한 답변 스타일이 그대로 옮겨오기 쉬우니, 오늘 배운 길이 보정 평가를 함께 쓰면 "진짜 성능"과 "말투"를 구분해 볼 수 있습니다.
