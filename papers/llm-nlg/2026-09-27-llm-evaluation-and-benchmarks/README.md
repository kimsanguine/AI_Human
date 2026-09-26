# Daily AI Paper Recommendations

> **Date:** 2026-09-27
> **Module:** Module 6: LLM for Natural Language Generation
> **Topic:** LLM Evaluation and Benchmarks

---

## Paper 1 (Classic): Measuring Mathematical Problem Solving With the MATH Dataset
- **Authors:** Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
- **Year:** 2021
- **arXiv:** https://arxiv.org/abs/2103.03874
- **PDF:** [./math-dataset-hendrycks-2021.pdf](./math-dataset-hendrycks-2021.pdf)
- **Citation Count:** approx. 2,500+

### 요약
고등학교 수학 경시대회(AMC 10/12, AIME 등)에서 가져온 12,500개 문제로 구성된 MATH 벤치마크를 제안한다. 모든 문제에 단계별 풀이(LaTeX)가 포함되어 있고, 7개 과목과 5단계 난이도로 나뉜다. 발표 당시 대형 Transformer도 정확도가 한 자릿수에 머물러, "모델 크기만 키워서는 수학 추론이 해결되지 않는다"는 문제를 제기했다.

### 핵심 기여
- 최종 답만이 아니라 단계별 풀이까지 포함한 대규모 경시 수학 벤치마크 구축
- 과목(대수·기하·정수론 등) × 난이도(Level 1~5) 구조로 약점을 세밀하게 진단할 수 있는 설계
- 수학 사전학습용 보조 데이터셋 AMPS를 함께 공개하고, 스케일링만으로는 성능 향상이 더디다는 실험 결과 제시

### 이 논문이 중요한 이유
GSM8K가 "초등 산수 서술형"이라면 MATH는 "경시 수준 추론"을 재는 표준 지표다. o1, DeepSeek-R1 같은 추론 모델의 성과가 MATH(와 그 부분집합 MATH-500) 점수로 보고되기 때문에, 추론 모델 논문과 모델 카드를 읽으려면 이 벤치마크가 무엇을 측정하고 무엇을 측정하지 못하는지 알아야 한다. 정답이 명확해 자동 채점이 가능하다는 점은 이후 RLVR(검증 가능한 보상 기반 강화학습)의 데이터 원천이 되기도 했다.

### 사전 지식
- GPT 계열 언어모델의 few-shot 평가 방식
- 정답 일치(exact match) 채점과 답 정규화(예: `\boxed{}` 추출)의 개념
- MMLU(Hendrycks et al., 2020) 등 멀티태스크 벤치마크의 기본 구조

### 관련 논문
- [Training Verifiers to Solve Math Word Problems / GSM8K (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168)
- [Let's Verify Step by Step / PRM800K (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050)

### 실무 적용
사내 모델·프롬프트를 비교할 때 "과목 × 난이도" 매트릭스로 결과를 쪼개 보는 방식은 도메인 평가셋 설계에 그대로 쓸 수 있다. 또한 정답이 명확한 과제는 자동 채점기(verifier)를 붙여 회귀 테스트와 강화학습 보상으로 동시에 활용할 수 있다는 점을 보여준다.

---

## Paper 2 (Classic): Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge
- **Authors:** Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord
- **Year:** 2018
- **arXiv:** https://arxiv.org/abs/1803.05457
- **PDF:** [./arc-challenge-clark-2018.pdf](./arc-challenge-clark-2018.pdf)
- **Citation Count:** approx. 2,500+

### 요약
초등~중학교 과학 시험 객관식 문제 7,787개로 구성된 ARC 데이터셋을 제안한다. 핵심은 단순 검색(retrieval)이나 단어 동시출현(co-occurrence) 기반 방법이 모두 틀린 문제만 모은 **Challenge Set**을 따로 분리했다는 점이다. 당시 SOTA QA 모델들이 Challenge Set에서 무작위 추측 수준을 크게 넘지 못함을 보여주며, "진짜 추론"이 필요한 문제를 측정하는 방법을 제시했다.

### 핵심 기여
- 기존 베이스라인이 틀리는 문제만 걸러내는 "적대적 필터링" 방식으로 Easy/Challenge 분할 설계
- 문제 풀이에 필요한 배경지식을 담은 1,400만 문장 규모의 ARC Corpus 공개
- 표면적 패턴 매칭으로 풀리는 QA와 추론이 필요한 QA를 구분하는 평가 관점 제시

### 이 논문이 중요한 이유
ARC-Challenge는 HellaSwag·MMLU와 함께 Open LLM Leaderboard 등에서 오랫동안 기본 지표로 쓰였다. 더 중요한 것은 "쉬운 모델이 푸는 문제는 제외한다"는 필터링 아이디어로, 이는 HellaSwag의 Adversarial Filtering, GPQA의 "구글로 못 푸는 문제", HLE의 "최신 모델이 틀리는 문제만 채택"까지 이어지는 벤치마크 설계 원칙의 뿌리다.

### 사전 지식
- 객관식 QA 평가(정확도, 선택지별 로그우도 비교) 방식
- 정보 검색(IR) 기반 QA와 신경망 기반 독해 모델의 차이
- 데이터셋 편향(annotation artifact)과 지름길 학습(shortcut learning) 개념

### 관련 논문
- [HellaSwag: Can a Machine Really Finish Your Sentence? (Zellers et al., 2019)](https://arxiv.org/abs/1905.07830)
- [Can a Suit of Armor Conduct Electricity? OpenBookQA (Mihaylov et al., 2018)](https://arxiv.org/abs/1809.02789)

### 실무 적용
자체 평가셋을 만들 때, 현재 운영 중인 베이스라인(예: 소형 모델 또는 키워드 검색)이 이미 맞히는 문항은 걸러내고 "어려운 부분집합"을 따로 관리하면 모델 교체 시 변별력을 확보할 수 있다. RAG 시스템에서도 "검색만으로 답이 나오는 질문"과 "추론이 필요한 질문"을 분리해 평가하는 데 응용된다.

---

## Paper 3 (Recent): Humanity's Last Exam
- **Authors:** Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, ... , Summer Yue, Alexandr Wang, Dan Hendrycks (Center for AI Safety & Scale AI, 1,000+ contributors)
- **Year:** 2025 (Nature, 2026)
- **arXiv:** https://arxiv.org/abs/2501.14249
- **PDF:** [./humanitys-last-exam-phan-2025.pdf](./humanitys-last-exam-phan-2025.pdf)
- **Citation Count:** approx. 500+ (빠르게 증가 중)

### 요약
MMLU 등 기존 벤치마크에서 최신 LLM이 90% 이상을 기록하며 변별력을 잃자, 전 세계 전문가들이 출제한 2,500개의 최고난도 문제로 HLE를 구축했다. 수학·인문학·자연과학 등 수십 개 분야에 걸친 멀티모달 문제이며, 정답은 명확하고 검증 가능하지만 인터넷 검색으로는 빠르게 찾을 수 없도록 설계되었다. 출제 과정에서 최신 모델이 맞히는 문제는 탈락시켜, 발표 시점 프런티어 모델들의 정확도가 매우 낮았고 보정(calibration) 오류도 컸다.

### 핵심 기여
- 프런티어 모델로 사전 검증 → 모델이 틀린 문제만 전문가 리뷰로 넘기는 다단계 출제 파이프라인
- 객관식/단답형 자동 채점 구조로 대규모·재현 가능한 평가 가능
- 정확도와 함께 모델의 자신감 보정 오차(calibration error)를 핵심 지표로 보고
- 벤치마크 과적합 방지를 위한 비공개(held-out) 문항 세트 운영

### 이 논문이 중요한 이유
2025~2026년 프런티어 모델 발표에서 거의 빠지지 않는 지표가 HLE다. MMLU → MMLU-Pro → GPQA → HLE로 이어지는 "난이도 경쟁"의 현재 최전선을 보여주며, 동시에 "학술 시험형 벤치마크의 마지막"을 자처함으로써 이후 평가가 에이전트·실제 업무 기반으로 이동해야 함을 시사한다. 또한 자신감 보정 지표는 환각(hallucination) 문제를 수치로 다루는 좋은 예시다.

### 사전 지식
- MMLU/MMLU-Pro, GPQA 등 지식·추론 벤치마크의 계보
- 모델 보정(calibration)과 Expected Calibration Error 개념
- 벤치마크 오염(contamination)과 비공개 테스트셋의 필요성

### 관련 논문
- [GPQA: A Graduate-Level Google-Proof Q&A Benchmark (Rein et al., 2023)](https://arxiv.org/abs/2311.12022)
- [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark (Wang et al., 2024)](https://arxiv.org/abs/2406.01574)

### 실무 적용
모델이 "모른다"고 말해야 할 때 과도하게 자신하는지를 측정하는 보정 지표는, 고객 응대·전문 상담 AI에서 에스컬레이션(사람에게 넘기기) 기준을 설계하는 데 직접 활용된다. 또한 "현재 모델이 틀리는 사례를 수집해 평가셋을 계속 갱신하는" HLE의 방식은 운영 중인 AI 제품의 실패 로그 기반 평가셋 운영(eval flywheel)과 같은 구조다.

---

## 추천 읽기 순서
1. **ARC (2018)** — "쉬운 방법이 푸는 문제를 걸러낸다"는 벤치마크 설계 원칙을 먼저 잡는다.
2. **MATH (2021)** — 검증 가능한 정답 + 단계별 풀이 구조가 왜 추론 평가와 학습(RLVR) 모두에 중요한지 이해한다.
3. **Humanity's Last Exam (2025)** — 두 원칙(적대적 필터링 + 검증 가능한 정답)이 프런티어 수준에서 어떻게 결합되는지 확인한다.

## 핵심 테이크어웨이
- 좋은 벤치마크의 공통 설계 원칙은 "현재 모델이 틀리는 문제만 남긴다(적대적 필터링)"와 "정답을 자동으로 검증할 수 있다"의 두 가지다.
- 벤치마크 수명은 점점 짧아진다. ARC → MATH → HLE로 가며 포화 주기가 수년에서 수개월로 줄고 있다.
- 정확도만큼 보정(calibration)이 중요하다. "틀린 답을 확신하는 모델"은 실제 제품에서 가장 위험하다.
- 실무에서는 공개 벤치마크보다, 우리 제품의 실패 사례로 계속 갱신되는 자체 평가셋이 최종 의사결정 기준이 되어야 한다.

## 다음 토픽과의 연결
다음 토픽인 **Efficient LLM (Quantization & Distillation)**에서는 모델을 줄였을 때 무엇을 잃는지 측정해야 한다. 흥미롭게도 양자화·증류의 성능 저하는 쉬운 벤치마크보다 MATH·HLE처럼 어려운 다단계 추론 과제에서 먼저 드러나는 경우가 많으므로, 오늘 다룬 "어려운 부분집합" 평가가 경량화 품질 검증의 핵심 도구가 된다.

---

### 참고: 논문 선정 노트
본 커리큘럼은 7번째 순환(cycle) 중이며, 이전 순환에서 사용한 논문(MMLU·HumanEval, HELM·MT-Bench, GLUE·BIG-bench, HellaSwag·GSM8K, SuperGLUE·TruthfulQA / 최신: MMLU-Pro·LiveBench·Arena-Hard·LiveCodeBench·Chatbot Arena)과 중복을 피하기 위해 MATH·ARC(클래식)와 Humanity's Last Exam(최신)을 새로 선정했다.
