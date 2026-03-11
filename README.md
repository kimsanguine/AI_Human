# AI Human — 초급자를 위한 AI Engineer 100일 커리큘럼

![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue)
![Last Commit](https://img.shields.io/github/last-commit/kimsanguine/AI_Human)
![News](https://img.shields.io/badge/AI%20뉴스-매일%20자동%20업데이트-brightgreen)

> **"코드를 처음 치는 사람도, 100일 뒤에는 RAG 파이프라인을 만든다."**

비전공자·초급자가 AI Engineer로 성장하기 위한 실전 커리큘럼입니다.
Python 기초부터 LLM, 프롬프트 엔지니어링, RAG까지 — 17주 동안 단계별로 학습합니다.

매일 오전, AI가 현재 학습 챕터와 연관된 뉴스를 자동으로 큐레이션하여 **학습과 업계 동향을 동시에** 따라갈 수 있습니다.

---

## 이런 분을 위해 만들었습니다

| 대상 | 기대 효과 |
|------|----------|
| AI에 관심 있는 비전공자 | Python부터 차근차근 시작 |
| PM/기획자가 AI를 이해하고 싶을 때 | 기술의 원리를 체험하며 커뮤니케이션 역량 향상 |
| AI 직무 전환을 준비하는 분 | KDT 수준의 체계적 커리큘럼으로 포트폴리오 구축 |
| 현업 개발자의 AI 리스킬링 | ML/DL → LLM → RAG로 빠른 확장 |

---

## 커리큘럼

```
Phase 1: 기초 체력        Phase 2: AI 핵심         Phase 3: 실전 응용
━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━     ━━━━━━━━━━━━━━━━━━━
W1  Python 기초           W5-9  NLP & 음성         W13 프롬프트 엔지니어링
W2  전처리 & 시각화        W10-11 TTS & STT        W14-15 LangChain
W3-4 ML & DL             W12 LLM                  W16-17 RAG
```

| 챕터 | 주제 | 기간 | 핵심 키워드 |
|------|------|------|-----------|
| Ch01 | Python 프로그래밍 | W1 | 환경 설정, 기초 문법, OOP |
| Ch02 | 전처리 및 시각화 | W2 | NumPy, Pandas, Matplotlib |
| Ch03 | 머신러닝과 딥러닝 | W3~W4 | Scikit-learn, PyTorch |
| Ch04 | 자연어 및 음성 데이터 | W5~W9 | NLP, 음성 처리 |
| Ch05 | TTS와 STT 모델 개발 | W10~W11 | 음성 합성, 음성 인식 |
| Ch06 | 거대 언어 모델 | W12 | GPT, Claude, LLM 활용 |
| Ch07 | 프롬프트 엔지니어링 | W13 | 프롬프트 설계, 최적화 |
| Ch08 | LangChain 활용 | W14~W15 | 체이닝, 메모리, 도구 |
| Ch09 | RAG | W16~W17 | 검색 증강 생성, 벡터 DB |

---

## 프로젝트 구조

```
AI_Human/
├── lectures/                    # 챕터별 강의 자료
│   ├── ch01-python/
│   ├── ch02-preprocessing/
│   ├── ...
│   └── ch09-rag/
├── news/
│   ├── daily/                   # 매일 AI 뉴스 브리핑
│   └── weekly/                  # 주간 다이제스트
├── curriculum/
│   └── mapping.json             # 챕터별 키워드 & 일정 매핑
├── LICENSE
└── README.md
```

---

## AI 뉴스 자동화 시스템

매일 오전 8:00 (KST, 월~금) 자동으로 실행됩니다.

```
[뉴스 검색] → [현재 챕터 키워드 매칭] → [daily/.md 생성]
                                         ↓
                                    [Slack 포스팅]
                                         ↓
                                    [Telegram 발송]
                                         ↓
                              [금요일: 주간 다이제스트]
```

현재 학습 중인 챕터의 키워드를 기반으로 관련 뉴스를 우선 선별하여, **수업에서 배운 개념이 실제 업계에서 어떻게 쓰이는지** 자연스럽게 연결합니다.

---

## 시작하기

```bash
# 1. 레포 클론
git clone https://github.com/kimsanguine/AI_Human.git

# 2. 오늘의 뉴스 확인
cat news/daily/$(date +%Y-%m-%d).md

# 3. 현재 챕터 강의 자료 열기
ls lectures/ch01-python/
```

---

## 저작권

© 2026 김생근 (Sanguine Kim) | CC BY-NC 4.0

| 콘텐츠 | 라이선스 | 비고 |
|--------|---------|------|
| `lectures/` 강의 자료 | CC BY-NC 4.0 | 교육 목적 자유 이용, 상업적 이용 제한 |
| `news/` 큐레이션 | CC BY-NC 4.0 | 원본 기사 저작권은 해당 매체에 귀속 |
| `curriculum/` 설정 | MIT | 자유 사용 |

교육·학술 목적으로 자유롭게 이용할 수 있습니다.

---

**김생근** · AI Human 튜터 · [GitHub](https://github.com/kimsanguine) · [LinkedIn](https://linkedin.com/in/sanguinekim)

AI B2B/B2C SaaS CPO, 20년 프로덕트 매니저.
LINE에서 1억명의 월렛 추천시스템을, Kakao에서 광고플랫폼을, ESTsoft에서 AI 더빙·아바타 서비스를 만들었습니다.
지금은 Agentic AI로 PM의 일하는 방식을 재설계하고 있습니다.
