# 4대 Gap 해소 및 Phase 3-5 개선 방향

> KDT AI Human 5기 — 20주 5Phase 커리큘럼 개선 설계
> 작성일: 2026-03-05
> 작성자: Ethan (with Claude)
> 상태: Gap ② 확정 완료

---

## 0. 전제 조건

| 항목 | 확정 내용 |
|------|----------|
| 총 기간 | 20주 고정 |
| Phase 구조 | 5Phase (기존 4Phase → 5Phase) |
| Phase 1-2 | 기존 교안 유지 |
| 개선 범위 | Phase 3-5에서 4대 Gap 해소 |
| 강의 운영 | **Ethan 전 과정 주관**, 직원 on-call 호출 가능 |

### Phase 구조

| Phase | 주차 | 주제 |
|-------|------|------|
| Phase 1 | W1-5 | Python 기초 + AI 도구 활용 |
| Phase 2 | W6-10 | ML/DL/NLP 핵심 |
| **Phase 3** | **W11-13** | **LLM + PE + 음성AI + LoRA** |
| **Phase 4** | **W14-17** | **LangChain + RAG + Agent + 서비스 프로토타이핑** |
| **Phase 5** | **W18-20** | **캡스톤 프로젝트** |

---

## 1. 비교표: 기존 커리큘럼 vs 개선안 (4대 Gap)

### Gap ① LLM Fine-tuning — QLoRA

| 구분 | 기존 커리큘럼 | 개선안 |
|------|-------------|--------|
| 분량 | BERT Fine-tuning 3일 (W9-10) | +5일 (W13 전체) |
| 범위 | BERT 분류 FT만 | LoRA/QLoRA + PEFT + HuggingFace Hub 배포 |
| 실습 | BERT 감성분석 FT | sLLM(Gemma-2B 등) QLoRA FT + 배포 |
| 도구 | HuggingFace Transformers | + PEFT, BitsAndBytes, Unsloth |
| 배치 | Phase 2 (W9-10) | **Phase 3 W13** (LLM 이해 후 바로 FT) |

### Gap ② LLMOps — ✅ Streamlit 기반 확정

| 구분 | 기존 커리큘럼 | 개선안 |
|------|-------------|--------|
| 분량 | Flask+ONNX 1일(W7D4) + Streamlit/Gradio 1일(W19) | **W17 전체(5일) + W19 MLflow 2h** |
| 범위 | Flask 모델서빙 맛보기 + 데모UI 맛보기 | **Streamlit 심화 + Gradio + 클라우드 배포 + API 개념 소개** |
| 실습 | Flask REST API 1개 + Streamlit 데모 1개 | RAG 챗봇 UI, 모델 데모앱, Streamlit Cloud/HF Spaces 배포 |
| 도구 | Flask, ONNX, Streamlit | **Streamlit, Gradio, Streamlit Cloud, HuggingFace Spaces** |
| 배치 | Phase 2(W7) + Phase 4(W19) | **Phase 4 W17** (RAG+Agent 완료 후 서비스화) |
| ~~FastAPI/Docker~~ | ~~없음~~ | ~~❌ 제외~~ — "다음 단계" 소개만 (0.5일) |

> **변경 사유:** Ethan이 전 과정을 직접 가르치는 구조에서, FastAPI(비동기+REST설계)와 Docker(컨테이너 인프라)는 학습-교수 준비 부담이 과도. Streamlit은 Python만 알면 30분 내 첫 앱 제작 가능하여 현실적. 채용시장에서도 주니어에게는 "동작하는 데모"가 Docker 경험보다 중요.

### Gap ③ Git/GitHub

| 구분 | 기존 커리큘럼 | 개선안 |
|------|-------------|--------|
| 분량 | GitHub 업로드 안내 1일 (W20) | **+6시간 분산 삽입** |
| 범위 | 프로젝트 결과물 업로드만 | init/add/commit/push + Branch/PR/Review |
| 실습 | 업로드 1회 | Phase 전반에 걸쳐 점진적 사용 |
| 배치 | Phase 4 마지막(W20) | **W1-2 기초 → Phase 3-4 실전 사용 → W18 심화** |

**점진적 삽입 계획:**

| 시점 | 내용 | 시간 |
|------|------|------|
| W1-2 (Phase 1) | Git 기초: init, add, commit, push + GitHub 계정 | 2h |
| W11 (Phase 3) | 실습 코드 Git 관리 시작, .gitignore | 1h |
| W14 (Phase 4) | LangChain 프로젝트 Git 기반 관리 | 실전 적용 |
| W18 (Phase 5) | Branch, PR, Code Review, 팀 협업 | 3h |

### Gap ④ Agent Framework 심화

| 구분 | 기존 커리큘럼 | 개선안 |
|------|-------------|--------|
| 분량 | Ethan Time 체험 6h (W14, W16) | **W16 풀위크(5일)** |
| 범위 | Agent 개념 + CrewAI/MCP 맛보기 | LangGraph 상태머신 + MCP + CrewAI 멀티에이전트 |
| 실습 | 데모 수준 체험 | 직접 Agent 설계 → 구현 → 평가 |
| 배치 | Ethan Time 보충 | **Phase 4 W16** (LangChain+RAG 후 자연 확장) |

### 종합 비교

| Gap | 기존 | 개선 | 증가분 |
|-----|------|------|--------|
| ① QLoRA | 3일 (BERT FT) | +5일 (W13) | **BERT→sLLM FT** |
| ② LLMOps | 2일 (Flask+Streamlit) | +5일 (W17) | **Streamlit 심화+배포** |
| ③ Git | 1일 (업로드) | +6h (분산) | **점진적 Git 습관** |
| ④ Agent | 6h (체험) | +5일 (W16) | **설계→구현→평가** |

---

## 2. Gap ① 상세: LLM Fine-tuning — QLoRA

### Q. 왜 필요한가?

채용공고 분석 결과 LoRA/QLoRA가 Tier 2 스킬(55%+ 언급)로 올라왔고, 기존 커리큘럼은 BERT Fine-tuning만 다루고 있어 sLLM 시대의 실무 격차가 큼.

### Q. 무엇을 다루는가?

| 일차 | W13 내용 |
|------|----------|
| D1 | LoRA 이론: Low-Rank Adaptation 수학적 직관 + PEFT 라이브러리 |
| D2 | QLoRA 실습: 4bit 양자화 + Gemma-2B 감성분류 FT |
| D3 | 학습 최적화: 하이퍼파라미터 튜닝 + WandB 실험 추적 |
| D4 | Unsloth 활용: 고속 FT + 다양한 태스크 적용 |
| D5 | HuggingFace Hub 배포 + 미니 프로젝트 |

### Q. Ethan의 역할은?

Ethan이 직접 강의. LoRA 이론은 논문 기반 설명 + Colab 라이브 코딩. 직원은 GPU 환경 트러블슈팅 시 호출.

---

## 3. Gap ② 상세: LLMOps — Streamlit 기반 서비스 프로토타이핑 ✅ 확정

### Q. 왜 Streamlit인가?

| 기준 | FastAPI+Docker | Streamlit |
|------|---------------|-----------|
| Ethan 학습 부담 | 높음 (비동기+REST+컨테이너) | **낮음** (Python만으로 즉시 시작) |
| 수강생 학습곡선 | 가파름 (백엔드 지식 필요) | **완만** (30분 내 첫 앱) |
| 포트폴리오 효과 | 높음 (프로덕션급) | **충분** (동작하는 데모 링크) |
| 채용시장 주니어 요구 | 40% (Docker), 35% (MLOps) | "데모 가능" > "인프라 경험" |
| 무료 배포 옵션 | 없음 (서버 필요) | **있음** (Streamlit Cloud, HF Spaces) |

### Q. W17 일별 계획은?

| 일차 | W17 내용 |
|------|----------|
| D1 | Streamlit 핵심: 컴포넌트, 레이아웃, 세션 상태, 캐싱 |
| D2 | RAG 챗봇 UI 만들기: W15 RAG 파이프라인 + Streamlit 연동 |
| D3 | Gradio로 모델 데모앱: 인터페이스, 블록, 이벤트 핸들링 |
| D4 | 클라우드 배포: Streamlit Cloud + HuggingFace Spaces 실전 배포 |
| D5 AM | API 개념 소개: "FastAPI/Docker란 이런 것" (맛보기, 코드 데모만) |
| D5 PM | 미니 해커톤: 팀별 AI 서비스 프로토타입 만들기 |

### Q. FastAPI/Docker는 완전히 버리는가?

아니요. **W17D5 오전에 "다음 단계" 소개**로 다룹니다:
- REST API 개념 (3분 설명)
- FastAPI Hello World 데모 (Ethan 라이브 코딩)
- Docker란? 컨테이너 개념 그림 설명
- "입사 후 이걸 배우게 됩니다" 로드맵 제시

이렇게 하면 수강생이 "들어본 적은 있다" 수준이 되고, 캡스톤에서 관심 있는 팀은 도전할 수 있습니다.

### Q. MLflow는?

W19(캡스톤 2주차)에 2시간 맛보기로 유지:
- 실험 로깅 (파라미터, 메트릭)
- 모델 레지스트리 개념
- Streamlit 앱과 연동해서 실험 결과 대시보드

### Q. Ethan의 역할은?

Ethan이 직접 강의. Streamlit은 1주 내 충분히 준비 가능. Gradio도 HuggingFace 생태계와 자연스럽게 연결. 직원은 배포 환경 이슈 시 호출.

---

## 4. Gap ③ 상세: Git/GitHub 점진적 삽입

### Q. 왜 점진적인가?

Git을 한 번에 몰아서 가르치면 "도구 학습"에 그치고, 실제 프로젝트에서 안 씀. Phase 전반에 걸쳐 자연스럽게 사용하게 해야 습관이 됨.

### Q. 구체적 삽입 계획은?

**W1-2 (Phase 1, 2h):**
- Git init, add, commit, push
- GitHub 계정 생성 + 첫 리포 만들기
- "오늘 실습 코드를 GitHub에 올려보세요" 과제

**W11 (Phase 3, 1h):**
- .gitignore (API 키, 모델 파일 제외)
- commit 메시지 컨벤션
- "이번 주 LLM 실습 전부 Git으로 관리"

**W14-17 (Phase 4, 실전 적용):**
- 모든 프로젝트를 Git 기반으로 진행
- 별도 시간 할당 없이 자연스러운 사용

**W18 (Phase 5, 3h):**
- Branch 전략 (main/develop/feature)
- Pull Request 작성법
- Code Review 실습 (팀원 코드 리뷰)
- GitHub Issues로 태스크 관리

### Q. Ethan의 역할은?

Ethan이 직접 시범. Git 기초는 PM 경험에서 이미 익숙한 영역. 심화(Branch/PR)는 3시간이면 충분히 전달 가능. 직원은 Git 충돌 해결 등 트러블슈팅 시 호출.

---

## 5. Gap ④ 상세: Agent Framework 심화

### Q. 왜 풀위크인가?

기존 6시간 체험으로는 "Agent가 뭔지 안다" 수준. 채용시장에서 Multi-agent(30%+)와 Agent 설계 능력을 요구하는 추세. W16 풀위크로 "직접 만들어봤다"까지 도달해야 함.

### Q. W16 일별 계획은?

| 일차 | W16 내용 |
|------|----------|
| D1 | Agent 아키텍처: ReAct, Plan-and-Execute, 상태머신 패턴 |
| D2 | LangGraph 실습: 상태 정의 → 노드/엣지 → 조건부 라우팅 |
| D3 | MCP (Model Context Protocol): 외부 도구 연동 에이전트 |
| D4 | CrewAI 멀티에이전트: 역할 분담, 협업 워크플로우 |
| D5 | 미니 프로젝트: 팀별 Agent 설계 → 구현 → 발표 |

### Q. Ethan의 역할은?

Ethan이 직접 강의 (✅ 확정). Agent/MCP/CrewAI는 Ethan의 현재 주력 분야(Agentic AI 제품 리딩)이므로 실무 경험 기반 강의 가능. 직원은 복잡한 LangGraph 디버깅 시 호출.

---

## 6. Phase 3-5 주차별 배치 종합

| 주차 | 메인 주제 | Gap 요소 |
|------|----------|----------|
| **W11** | LLM 기초 + Prompt Engineering | ③ Git 실전 적용 시작 (1h) |
| **W12** | 음성 AI (STT/TTS) + PE 심화 | — |
| **W13** | **LoRA/QLoRA Fine-tuning** | **① QLoRA 풀위크** |
| **W14** | LangChain 기초 + 체인 설계 | ③ Git 기반 프로젝트 관리 |
| **W15** | RAG 파이프라인 | ③ Git 기반 프로젝트 관리 |
| **W16** | **Agent Framework 심화** | **④ Agent 풀위크** |
| **W17** | **서비스 프로토타이핑 (Streamlit)** | **② LLMOps (Streamlit 기반)** |
| **W18** | 캡스톤 1주차: 기획+설계+개발 시작 | ③ Git 심화 (Branch/PR/Review, 3h) |
| **W19** | 캡스톤 2주차: 개발+테스트+배포 | ② MLflow 맛보기 (2h) |
| **W20** | 캡스톤 3주차: 발표+포트폴리오 | — |

---

## 7. 확정 사항 및 남은 판단

| # | 항목 | 상태 | 내용 |
|---|------|------|------|
| ① | 20주 5Phase 구조 | ✅ 확정 | Phase 3(W11-13), Phase 4(W14-17), Phase 5(W18-20) |
| ② | Ethan 전 강의 주관 | ✅ 확정 | 직원은 on-call 호출 방식 |
| ③ | Gap② Streamlit 기반 | ✅ 확정 | FastAPI/Docker → Streamlit+Gradio+클라우드 배포 |
| ④ | GPU 환경 | ❓ 확인 필요 | Colab Pro? 학원 서버? (QLoRA 실습용) |
| ⑤ | Phase 1 Git 삽입 | ❓ 확인 필요 | W1-2에 2시간 삽입 가능한지 |
| ⑥ | 캡스톤 팀 구성 | ❓ 확인 필요 | 몇 명/팀? 주제 자유? |
| ⑦ | NDA/데이터 범위 | ❓ 확인 필요 | 실습에 외부 데이터 사용 가능 범위 |

---

## 8. 다음 단계

1. 남은 판단사항(④⑤⑥⑦) 확정
2. Phase 3 일별 교안 작성 시작 (W11 → W12 → W13)
3. Phase 4 일별 교안 작성 (W14 → W15 → W16 → W17)
4. Phase 5 일별 교안 작성 (W18 → W19 → W20)

> 교안 작성 우선순위: W13(QLoRA) → W16(Agent) → W17(Streamlit) → 나머지 순서 제안
> 이유: Gap 해소 핵심 주차를 먼저 확정해야 나머지 주차 연결이 자연스러움
