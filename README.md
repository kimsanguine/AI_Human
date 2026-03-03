# AI Human Daily Brief

AI Human 100일 과정 수강생을 위한 매일 AI 뉴스 큐레이션 & 커리큘럼 연결 자동화 시스템

## 구조

```
AI_Human/
├── daily/           # 매일 AI 뉴스 브리핑 (YYYY-MM-DD.md)
├── weekly/          # 주간 다이제스트 (week-XX.md)
├── curriculum/      # 커리큘럼 ↔ 뉴스 매핑 데이터
│   └── mapping.json # 챕터별 키워드 & 일정 매핑
└── README.md
```

## 커리큘럼 (100일 과정)

| 챕터 | 주제 | 기간 |
|------|------|------|
| Ch01 | Python 프로그래밍 | W1 |
| Ch02 | Python 전처리 및 시각화 | W2 |
| Ch03 | 머신러닝과 딥러닝 | W3~W4 |
| Ch04 | 자연어 및 음성 데이터 활용 | W5~W9 |
| Ch05 | TTS와 STT 모델 개발 | W10~W11 |
| Ch06 | 거대 언어 모델 자연어 생성 | W12 |
| Ch07 | 프롬프트 엔지니어링 | W13 |
| Ch08 | Langchain 활용하기 | W14~W15 |
| Ch09 | RAG | W16~W17 |
| Ch10 | AI 기능 구현 실습 | W18~W20 |

## 자동화

매일 오전 8:00 (KST) Claude Scheduled Task가 실행되어:
1. AI 뉴스 웹 검색 → 커리큘럼 매칭
2. `daily/YYYY-MM-DD.md` 파일 생성
3. Slack `#ai-human-news` 채널 포스팅
4. Telegram 채널 발송

---

*Powered by Claude Cowork Scheduled Tasks*
