# AI Human 100일 과정

AI Human 100일 과정 강의 자료 + 매일 AI 뉴스 큐레이션 자동화 시스템

## 구조

```
AI_Human/
├── lectures/                    # 강의 자료
│   ├── ch01-python/
│   ├── ch02-preprocessing/
│   ├── ch03-ml-dl/
│   ├── ch04-nlp-speech/
│   ├── ch05-tts-stt/
│   ├── ch06-llm/
│   ├── ch07-prompt-engineering/
│   ├── ch08-langchain/
│   └── ch09-rag/
├── news/
│   ├── daily/                   # 매일 AI 뉴스 브리핑 (YYYY-MM-DD.md)
│   └── weekly/                  # 주간 다이제스트 (week-NN.md)
├── curriculum/
│   └── mapping.json             # 챕터별 키워드 & 일정 매핑
└── README.md
```

## 커리큘럼 (100일 과정)

| 챕터 | 주제 | 기간 | 폴더 |
|------|------|------|------|
| Ch01 | Python 프로그래밍 | W1 | `lectures/ch01-python/` |
| Ch02 | Python 전처리 및 시각화 | W2 | `lectures/ch02-preprocessing/` |
| Ch03 | 머신러닝과 딥러닝 | W3~W4 | `lectures/ch03-ml-dl/` |
| Ch04 | 자연어 및 음성 데이터 활용 | W5~W9 | `lectures/ch04-nlp-speech/` |
| Ch05 | TTS와 STT 모델 개발 | W10~W11 | `lectures/ch05-tts-stt/` |
| Ch06 | 거대 언어 모델 자연어 생성 | W12 | `lectures/ch06-llm/` |
| Ch07 | 프롬프트 엔지니어링 | W13 | `lectures/ch07-prompt-engineering/` |
| Ch08 | Langchain 활용하기 | W14~W15 | `lectures/ch08-langchain/` |
| Ch09 | RAG | W16~W17 | `lectures/ch09-rag/` |

## 뉴스 자동화

매일 오전 8:00 (KST, 월~금) Claude Scheduled Task가 실행되어:
1. AI 뉴스 웹 검색 → 현재 커리큘럼 챕터 매칭
2. `news/daily/YYYY-MM-DD.md` 파일 생성 및 커밋
3. Slack `#ai-human-news` 채널 포스팅
4. Telegram 채널 발송
5. 금요일마다 `news/weekly/week-NN.md` 주간 다이제스트 추가

---

*Powered by Claude Cowork Scheduled Tasks*
