---
name: ai-human-daily-brief
description: "AI Human 100일 과정 매일 AI 뉴스 큐레이션 → Slack + Telegram + GitHub 배포"
cron: "0 8 * * 1-5"
---

## 목적
AI Human 100일 과정 수강생을 위한 매일 AI 뉴스 큐레이션 및 멀티채널 배포를 수행한다.
커리큘럼 진도에 맞춘 뉴스를 선별하여 Slack, Telegram, GitHub에 자동 공유한다.

## 핵심 정보
- GitHub 레포: https://github.com/kimsanguine/AI_Human
- Slack 채널: #ai-human-news
- Telegram Bot Token: 8236282258:***REDACTED***
- Telegram Chat ID: 8595911950
- 과정 시작일: 2026-03-04 (수요일) — Day 1
- 커리큘럼 매핑 파일: ~/Documents/AI_Human/curriculum/mapping.json

## 커리큘럼 ↔ 일자 매핑 (평일 기준 Day 카운트)

Day 1~6: Ch01 Python 프로그래밍 → 키워드: Python AI, AI coding tools, VS Code AI, Python ecosystem
Day 7~13: Ch02 Python 전처리 및 시각화 → 키워드: data science, pandas, data visualization, AI data pipeline
Day 14~26: Ch03 머신러닝과 딥러닝 → 키워드: machine learning, deep learning, PyTorch, CNN, neural network, AutoML
Day 27~53: Ch04 자연어 및 음성 데이터 → 키워드: NLP, tokenizer, Transformer, BERT, embedding, attention mechanism, Korean NLP, speech data
Day 54~63: Ch05 TTS와 STT → 키워드: TTS, text to speech, STT, Whisper, voice synthesis, ElevenLabs, voice cloning
Day 64~70: Ch06 거대 언어 모델 → 키워드: LLM, GPT, Claude, Gemini, LLaMA, open source LLM, fine-tuning, Ollama
Day 71~78: Ch07 프롬프트 엔지니어링 → 키워드: prompt engineering, chain of thought, few-shot, system prompt
Day 79~90: Ch08 Langchain → 키워드: LangChain, AI agent, tool use, MCP, agentic AI, vector database, Pinecone, Chroma
Day 91~100: Ch09 RAG → 키워드: RAG, retrieval augmented generation, semantic search, embedding model, hybrid search, knowledge graph

## Phase별 뉴스 할당 (매일 5건 고정)
아래 할당 수를 반드시 준수한다. 카테고리 A는 커리큘럼 직결 뉴스, 카테고리 B는 AI 업계 동향/실무 사례다.

- Phase 1 (Day 1~13, Ch01~02): A(Python 생태계) **2건** + B(AI 업계 동향) **3건**
- Phase 2 (Day 14~26, Ch03): A(ML/DL 연구·기술) **3건** + B(AI 업계 동향) **2건**
- Phase 3 (Day 27~63, Ch04~05): A(NLP/음성 전문) **3건** + B(AI 업계 동향) **2건**
- Phase 4 (Day 64~100, Ch06~09): A(LLM/Agent 뉴스) **3건** + B(실무 적용 사례) **2건**

선별 후 A/B 건수를 확인하고, 비율이 맞지 않으면 검색을 추가 수행하여 조정한다.

## 실행 단계

### Step 1: 현재 Day 번호 계산
오늘 날짜에서 과정 시작일(2026-03-04)을 기준으로 경과한 평일 수를 계산하여 현재 Day 번호를 산출한다. 주말은 제외한다. Day 번호로 위 매핑에서 현재 챕터와 키워드를 결정한다.
만약 Day가 100을 초과하면 "과정 종료 후 — AI 전반 동향" 모드로 운영한다.

### Step 1.5: 최근 뉴스 중복 체크
~/Documents/AI_Human/news/daily/ 폴더에서 최근 5개의 .md 파일을 읽는다 (날짜 역순).
각 파일에서 뉴스 제목(### 으로 시작하는 줄)을 추출하여 "최근 다룬 뉴스 목록"을 만든다.
이 목록에 있는 뉴스는 오늘 선별 시 반드시 제외한다. 동일한 사건의 후속 보도(예: "OpenAI 투자 유치" → "OpenAI 투자 후속 영향")는 허용하되, 새로운 관점이 있을 때만 포함한다.
파일이 5개 미만이면 있는 만큼만 확인한다.

### Step 2: AI 뉴스 검색
WebSearch 도구를 사용하여 최소 4개의 검색을 수행한다. 검색 쿼리에는 반드시 **오늘 또는 이번 주 날짜**를 포함하여 최신 결과를 유도한다.

- 검색 1 (커리큘럼 연구/기술): "[현재 챕터 키워드 중 2~3개] news [이번 주 날짜 범위, 예: March 31 2026]"
  - allowed_domains 추천: techcrunch.com, nature.com, arxiv.org, technologyreview.com, venturebeat.com
- 검색 2 (AI 업계 동향): "AI news [오늘 요일 month day 2026]" 또는 "AI news this week [month] 2026"
  - allowed_domains 추천: techcrunch.com, theverge.com, wired.com, reuters.com, bloomberg.com
- 검색 3 (커리큘럼 보충): "[현재 챕터 키워드 중 다른 2~3개] breakthrough latest [month 2026]"
- 검색 4 (한국어 뉴스 보충): "[현재 챕터 한국어 키워드] AI 뉴스 2026" (한국 관련 뉴스 발굴용)

검색 결과가 부족하면 추가 검색을 수행한다 (최대 6개까지 허용).

총 **정확히 5건**의 뉴스를 선별한다. 5건보다 적거나 많으면 안 된다.
Step 1.5에서 만든 "최근 다룬 뉴스 목록"과 대조하여 중복을 제거한다.
각 뉴스에 대해 제목, 출처 URL(발행일 포함), 핵심 요약(2~3줄 한국어), 학습 연결 포인트를 작성한다.
뉴스 원문을 그대로 복사하지 않는다. 핵심만 자신의 언어로 요약한다.

### Step 2.5: 학습 연결 품질 기준
학습 연결을 작성할 때 아래 규칙을 반드시 따른다:
- **현재 챕터의 구체적 기술 개념을 1개 이상 명시**한다. (예: "합성곱 연산", "역전파", "활성화 함수", "손실 함수", "경사하강법" 등)
- 뉴스의 기술과 현재 학습 내용이 **어떻게 연결되는지 구체적으로 1문장**으로 설명한다.
- "큰 그림을 그려볼 수 있다", "체감할 수 있다" 같은 **모호한 표현을 금지**한다.
- 좋은 예: "Conformer는 CNN(지역 패턴 추출)과 Transformer(장거리 의존성 모델링)를 결합한 구조로, Ch03에서 배우는 합성곱 연산이 음성 데이터에서 어떻게 확장되는지 보여준다."
- 나쁜 예: "지금 배우는 신경망이 큰 그림으로 어떻게 발전하는지 느낄 수 있다."

### Step 3: 콘텐츠 생성
아래 마크다운 형식으로 Daily Brief를 생성한다:

```markdown
# AI Human Daily Brief — [YYYY-MM-DD] (요일) | Day [N]/100

**현재 진도:** [챕터명] — [세부 주제]

---

## TODAY'S TOP AI NEWS

### 1. [뉴스 제목]
**출처:** [미디어명 — YYYY-MM-DD](URL)

[핵심 요약 2~3줄, 한국어]

> **학습 연결:** [현재 배우고 있는 내용과 이 뉴스의 연결점]

### 2. [뉴스 제목]
**출처:** [미디어명 — YYYY-MM-DD](URL)

[핵심 요약 2~3줄]

> **학습 연결:** [연결점]

### 3. [뉴스 제목]
**출처:** [미디어명 — YYYY-MM-DD](URL)

[핵심 요약 2~3줄]

> **학습 연결:** [연결점]

### 4. [뉴스 제목]
**출처:** [미디어명 — YYYY-MM-DD](URL)

[핵심 요약 2~3줄]

> **학습 연결:** [연결점]

### 5. [뉴스 제목]
**출처:** [미디어명 — YYYY-MM-DD](URL)

[핵심 요약 2~3줄]

> **학습 연결:** [연결점]

---

## 오늘의 토론 질문
> "[현재 챕터 주제와 연결된 생각해볼 질문 1개]"

---
**김생근** | AI Human 튜터
AI B2B/B2C SaaS CPO, 20년 프로덕트 매니저. AI Dubbing·Avatar·Agentic AI 제품을 리딩하며 AI 네이티브 사고를 실무에 적용하고 있습니다.
```

### Step 4: GitHub 저장
생성된 콘텐츠를 ~/Documents/AI_Human/news/daily/YYYY-MM-DD.md 파일로 저장한다.
디렉토리가 없으면 생성한다 (mkdir -p ~/Documents/AI_Human/news/daily/).
저장 후 git commit과 push를 시도한다:
```bash
cd ~/Documents/AI_Human && git add news/ && git commit -m "daily: YYYY-MM-DD (Day N)" && git push origin main
```
git push가 실패하면 로그에 "GitHub push 실패 — 수동 push 필요"로 기록하고 계속 진행한다.

### Step 5: Slack 포스팅
Slack MCP 도구를 사용하여 #ai-human-news 채널에 포스팅한다.
1. slack_search_channels 도구로 "ai-human-news"를 검색하여 channel_id를 찾는다.
2. 찾은 channel_id로 slack_send_message를 호출한다.
3. Slack 메시지 형식 (mrkdwn):
```
*AI Human Daily Brief* — [날짜] | Day [N]/100
:pushpin: *현재 진도:* [챕터명]

:fire: *TODAY'S TOP AI NEWS*

*1. [뉴스 제목]* — 출처
[핵심 요약 2줄]
:bulb: _학습 연결: [연결점]_

*2. [뉴스 제목]* — 출처
[핵심 요약 2줄]
:bulb: _학습 연결: [연결점]_

*3. [뉴스 제목]* — 출처
[핵심 요약 2줄]

*4. [뉴스 제목]* — 출처
[핵심 요약 2줄]

*5. [뉴스 제목]* — 출처
[핵심 요약 2줄]

:dart: *오늘의 토론 질문*
> [질문]
```
Slack이 연결되지 않으면 스킵하고 결과에 "Slack 미연결"로 기록한다.

### Step 6: Telegram 발송
Bash에서 curl로 Telegram Bot API를 호출하여 메시지를 보낸다.
Telegram은 4096자 제한이 있으므로 간결하게 작성한다:
```bash
MESSAGE=$(cat <<'TELEGRAMEOF'
🤖 *AI Human Daily Brief* — [날짜] | Day N/100
📌 현재 진도: [챕터명]

🔥 *TODAY'S TOP AI NEWS*

1️⃣ *[제목]*
[요약 1줄]
💡 학습 연결: [연결점]

2️⃣ *[제목]*
[요약 1줄]
💡 학습 연결: [연결점]

3️⃣ *[제목]*
[요약 1줄]

4️⃣ *[제목]*
[요약 1줄]

5️⃣ *[제목]*
[요약 1줄]

🎯 *오늘의 토론 질문*
[질문]
TELEGRAMEOF
)

curl -s -X POST "https://api.telegram.org/bot8236282258:***REDACTED***/sendMessage" \
  -d chat_id=8595911950 \
  -d parse_mode=Markdown \
  --data-urlencode "text=$MESSAGE"
```
Telegram 발송 실패 시 결과에 "Telegram 발송 실패"로 기록한다.

### Step 7 (금요일만): 주간 다이제스트
오늘이 금요일이면 추가로 주간 다이제스트를 작성한다:
- ~/Documents/AI_Human/news/daily/ 폴더에서 이번 주(월~금) 파일들을 읽는다
- 이번 주 TOP 3 뉴스, 핵심 키워드, 다음 주 프리뷰를 정리한다
- ~/Documents/AI_Human/news/weekly/week-NN.md 파일로 저장한다
- Slack과 Telegram에도 주간 다이제스트를 별도 메시지로 발송한다

### Step 8: 실행 결과 요약
마지막에 채널별 성공/실패 상태를 출력한다:
- GitHub: 커밋 완료 / push 실패 (로컬 저장됨) / 저장 실패
- Slack: 포스팅 완료 / 미연결 / 발송 실패
- Telegram: 발송 완료 / 발송 실패

## 중요 제약사항
- 모든 콘텐츠는 한국어로 작성한다
- 뉴스 원문을 복사하지 않는다. 핵심만 요약하고 학습 연결 포인트를 추가한다
- 주말(토/일)에는 실행하지 않는다 (cron이 월~금 설정)
- Day 100 이후에도 AI 전반 뉴스 모드로 계속 운영한다

## 뉴스 신선도 기준
- **48시간 이내 뉴스를 최우선**으로 선별한다.
- 48시간 이내 뉴스가 부족할 경우 **7일 이내까지 허용**하되, 5건 중 최대 2건까지만 허용한다.
- 7일 이상 지난 뉴스는 선별하지 않는다.
- 출처 표기 시 **발행일을 반드시 포함**한다. 형식: `[미디어명 — YYYY-MM-DD](URL)`
  - 예시: `[TechCrunch — 2026-03-31](https://techcrunch.com/...)`
  - 정확한 발행일을 알 수 없으면 `[미디어명 — 2026년 3월 말](URL)` 형태로 근사치를 표기한다.

## 출처 품질 기준
- **원출처(1차 출처)를 우선 사용한다.** TechCrunch, Nature, arXiv, MIT Tech Review, The Verge, VentureBeat, Wired, Reuters, Bloomberg 등 직접 취재·발표한 미디어를 선호한다.
- **뉴스 집약/큐레이션 사이트**(Crescendo AI, AI Weekly, Radical Data Science, Digital Applied 등)는 원출처를 찾을 수 없을 때만 사용한다.
- 동일한 뉴스가 원출처와 집약 사이트 양쪽에 있으면, 반드시 원출처 URL을 사용한다.
- 5건 중 집약 사이트 출처는 **최대 1건**으로 제한한다.
