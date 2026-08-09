# Daily AI Paper Recommendations

> **Date:** 2026-08-10
> **Module:** Module 8: LangChain and LLM Orchestration
> **Topic:** AI Agents and Tool Use

---

## Paper 1 (Classic): ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs
- **Authors:** Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al.
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2307.16789](https://arxiv.org/abs/2307.16789)
- **PDF:** [./toolllm-qin-2023.pdf](./toolllm-qin-2023.pdf)
- **Citation Count:** approx. 1,000+

### 요약
오픈소스 LLM은 언어 태스크에는 강하지만 실제 도구(API)를 다루는 능력은 부족했다. ToolLLM은 RapidAPI Hub에서 수집한 49개 카테고리, 16,464개의 실제 RESTful API를 기반으로 데이터 구축, 모델 학습, 평가까지 아우르는 종합 도구 학습 프레임워크를 제시한다. 특히 DFSDT(Depth-First Search-based Decision Tree)로 다중 도구 호출 경로를 탐색하게 하여 LLaMA 기반 모델이 GPT 계열에 근접한 도구 사용 능력을 갖추도록 만들었다.

### 핵심 기여
- ToolBench: 단일·다중 도구 시나리오를 포함한 대규모 명령어-도구 사용 데이터셋을 ChatGPT로 자동 생성
- DFSDT: 기존 ReAct의 단선적 추론 한계를 넘어, 실패 시 백트래킹이 가능한 트리 탐색 기반 추론으로 성공률 향상
- ToolEval + ToolLLaMA: 자동 평가기와, API 리트리버를 결합해 미학습 API에도 일반화되는 오픈소스 도구 사용 모델 공개

### 이 논문이 중요한 이유
LangChain·에이전트 개발의 핵심은 "LLM이 외부 도구를 얼마나 안정적으로 호출하느냐"이다. ToolLLM은 함수 호출(function calling)과 도구 라우팅을 대규모·실전 API 관점에서 체계화한 대표 연구로, 오늘날 툴 사용 파인튜닝과 에이전트 벤치마크 설계의 출발점이 된다.

### 사전 지식
- ReAct 스타일 추론(Reasoning+Acting)과 function calling의 기본 개념
- RESTful API 호출 구조와 파라미터 스키마
- 명령어 튜닝(instruction tuning)과 리트리벌 기반 컨텍스트 주입

### 관련 논문
- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)
- [Gorilla: Large Language Model Connected with Massive APIs (Patil et al., 2023)](https://arxiv.org/abs/2305.15334)

### 실무 적용
사내 API가 수백 개인 B2B SaaS에서 에이전트가 올바른 API를 고르고 파라미터를 채우게 하려면, ToolLLM의 API 리트리버 + DFSDT 패턴이 직접적인 설계 참고가 된다. 실패 경로 백트래킹은 실제 프로덕션에서 도구 호출 실패율을 낮추는 재시도 전략으로 응용할 수 있다.

---

## Paper 2 (Classic): Voyager: An Open-Ended Embodied Agent with Large Language Models
- **Authors:** Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar
- **Year:** 2023
- **arXiv:** [https://arxiv.org/abs/2305.16291](https://arxiv.org/abs/2305.16291)
- **PDF:** [./voyager-wang-2023.pdf](./voyager-wang-2023.pdf)
- **Citation Count:** approx. 1,500+

### 요약
Voyager는 Minecraft에서 인간 개입 없이 스스로 탐험하고 기술을 익히며 새로운 것을 발견하는 최초의 LLM 기반 평생학습(lifelong learning) 에이전트다. GPT-4를 블랙박스로 질의하여 파라미터 파인튜닝 없이 동작하며, 자동 커리큘럼·스킬 라이브러리·반복적 프롬프팅이라는 세 축으로 능력을 점진적으로 축적한다. 특히 실행 가능한 코드를 스킬로 저장·재사용하는 구조가 핵심이다.

### 핵심 기여
- 자동 커리큘럼(automatic curriculum): 에이전트의 현재 상태에 맞춰 탐험을 극대화하는 목표를 스스로 생성
- 스킬 라이브러리: 검증된 행동을 실행 가능한 코드로 저장하고 임베딩 검색으로 재사용 → 조합적·해석 가능한 능력 확장
- 반복적 프롬프팅: 환경 피드백·실행 오류·자기검증(self-verification)을 프롬프트에 반영해 코드를 개선

### 이 논문이 중요한 이유
"코드를 스킬로 저장하고 재사용한다"는 아이디어는 오늘날 에이전트 메모리·툴 라이브러리 설계의 원형이다. 파인튜닝 없이 프롬프트/피드백 루프만으로 능력을 누적하는 접근은 실전 에이전트 아키텍처에 직접 영감을 준다.

### 사전 지식
- LLM 에이전트의 계획-실행-피드백 루프
- 임베딩 기반 검색(스킬/메모리 retrieval)
- 자기검증(self-verification)과 반성(reflection) 개념

### 관련 논문
- [Reflexion: Language Agents with Verbal Reinforcement Learning (Shinn et al., 2023)](https://arxiv.org/abs/2303.11366)
- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)

### 실무 적용
반복 업무를 처리하는 프로덕션 에이전트에서, 성공한 실행 결과를 "재사용 가능한 스킬(코드/워크플로우)"로 저장해 두면 이후 유사 작업의 처리 속도와 안정성이 크게 오른다. Voyager의 스킬 라이브러리는 에이전트용 장기 메모리·플레이북 캐시 설계의 실전 청사진이다.

---

## Paper 3 (Recent): OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments
- **Authors:** Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al.
- **Year:** 2024 (NeurIPS 2024)
- **arXiv:** [https://arxiv.org/abs/2404.07972](https://arxiv.org/abs/2404.07972)
- **PDF:** [./osworld-xie-2024.pdf](./osworld-xie-2024.pdf)
- **Citation Count:** approx. 500+

### 요약
OSWorld는 Ubuntu·Windows·macOS 등 실제 운영체제 환경에서 멀티모달 에이전트를 평가하는 최초의 확장 가능한 벤치마크다. 실제 웹·데스크톱 앱, OS 파일 입출력, 다중 앱 워크플로우를 포함한 369개 컴퓨터 태스크로 구성되며, 각 태스크는 초기 상태 설정과 실행 기반 평가 스크립트를 갖춰 재현 가능한 채점을 지원한다.

### 핵심 기여
- 실제 OS 환경에서 태스크 설정·실행·평가·상호작용 학습을 지원하는 확장형 벤치마크 인프라 제공
- 369개 실전 태스크와 실행 기반(execution-based) 평가로 신뢰도 높은 자동 채점 구현
- SOTA 에이전트의 심각한 한계 노출: 인간은 72.36% 이상 수행하는 반면 최고 모델은 12.24%에 그침

### 이 논문이 중요한 이유
"컴퓨터를 실제로 조작하는 에이전트"(computer-use)가 부상하는 흐름에서, OSWorld는 에이전트의 실제 능력을 가늠하는 표준 잣대다. 데모가 아닌 재현 가능한 실행 기반 평가를 제시해, 과대평가되기 쉬운 에이전트 성능을 냉정하게 측정하는 방법론을 정립했다.

### 사전 지식
- 멀티모달(스크린샷+텍스트) 에이전트의 인식-행동 루프
- 실행 기반 평가 vs. 정답 매칭 평가의 차이
- GUI 자동화(마우스/키보드 액션 스페이스)의 기본 개념

### 관련 논문
- [WebArena: A Realistic Web Environment for Building Autonomous Agents (Zhou et al., 2023)](https://arxiv.org/abs/2307.13854)
- [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (Yao et al., 2024)](https://arxiv.org/abs/2406.12045)

### 실무 적용
컴퓨터-유즈/RPA 성격의 에이전트 제품을 만든다면, OSWorld의 실행 기반 평가 설계를 사내 회귀 테스트에 이식해 릴리스마다 실제 태스크 성공률을 측정할 수 있다. 낮은 SOTA 성공률은 "완전 자율"보다 human-in-the-loop 설계가 현실적임을 시사한다.

---

## 추천 읽기 순서
1. **ToolLLM** — 에이전트의 기본기인 "도구/API 호출"을 대규모·실전 관점에서 이해
2. **Voyager** — 도구를 넘어 "스킬 축적과 평생학습"으로 에이전트 능력을 확장하는 아키텍처 학습
3. **OSWorld** — 이렇게 만든 에이전트를 실제 컴퓨터 환경에서 어떻게 냉정하게 평가하는지 확인

## 핵심 테이크어웨이
- 에이전트의 성능은 결국 **도구 선택·호출의 안정성**에서 갈린다 (ToolLLM: 리트리버 + 백트래킹 탐색).
- **성공 경험을 재사용 가능한 스킬로 저장**하는 구조가 장기적으로 에이전트를 강하게 만든다 (Voyager).
- 화려한 데모보다 **실행 기반·재현 가능한 평가**가 중요하며, 현재 SOTA도 실제 컴퓨터 태스크에선 인간에 크게 못 미친다 (OSWorld).

## 다음 토픽과의 연결
스킬 라이브러리(Voyager)와 실행 평가(OSWorld)는 모두 에이전트가 "무엇을 기억하고 어떻게 오래 유지하는가"의 문제로 이어진다. 다음 토픽인 **Memory and Long-Context Management**에서 MemGPT·Generative Agents를 통해 에이전트의 장기 메모리 아키텍처를 깊이 다룬다.
