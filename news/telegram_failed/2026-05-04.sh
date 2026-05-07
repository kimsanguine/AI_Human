#!/bin/bash
# AI Human Daily Brief — 2026-05-04 (Day 44/100)
MESSAGE=$(cat <<'MSG'
🤖 *AI Human Daily Brief — 2026-05-04 (월) | Day 44/100*

📚 *진도:* Ch04 NLP·음성 — Transformer/임베딩/토큰화 핵심 구간
📊 *구성:* 📰 5건

━━━━━━━━━━━━━━━━━━

*1. 📰 Mistral Medium 3.5 — 128B 덴스, 256K 컨텍스트, 오픈웨이트*
4월 29일 공개. Medium 3.1·Magistral·Devstral 2를 한 가중치로 통합, 요청별 reasoning_effort 토글. 4-GPU 자체 호스팅 가능.
🔗 https://docs.mistral.ai/models/model-cards/mistral-medium-3-5-26-04
💡 256K 컨텍스트 = 어텐션 시퀀스가 그만큼 길어짐. O(L²) 복잡도를 피하는 sliding window·GQA가 운영의 핵심.

*2. 📰 Qwen 3.6-Plus, Fireworks AI 단독 배포*
알리바바가 5월 초 Qwen 3.6-Plus를 Fireworks 단독 추론 채널로 공개. FireAttention 엔진으로 토큰 단가·지연시간 최적화.
🔗 https://x.com/Alibaba_Qwen/status/2039751581575659833
💡 같은 가중치라도 어디서 서빙하느냐가 ms 단위 latency를 가른다 — 학습 ≠ 운영.

*3. 📰 베이징, Meta-Manus $2B 인수 차단 — Singapore의 'AI 도피처' 모델 흔들*
NDRC가 4월 27일 차단, 5월 2일 Bloomberg 후속 분석. 중국계 AI 인재의 싱가포르 우회 경로 봉쇄.
🔗 https://www.bloomberg.com/news/newsletters/2026-05-02/beijing-blocks-meta-s-manus-deal-threatening-singapore-s-role-as-chinese-ai-hub
💡 NLP 토대 위 에이전트 = 정치·자본의 통제 대상. 한국어 NLP도 같은 지정학 프레임 안.

*4. 📰 Microsoft, Agent 365 + 365 Enterprise 7 GA — $99 'Frontier Worker Suite'*
5월 1일 정식 출시. Defender·Entra·Purview를 에이전트 객체로 확장. "통제되지 않은 에이전트 = 사내 이중간첩" 프레이밍.
🔗 https://venturebeat.com/ai/microsofts-agent-365-shifts-ai-agents-from-sandbox-tools-to-enterprise-grade
💡 Copilot 백엔드 BERT계 임베딩까지 Purview 데이터 분류 정책에 묶임.

*5. 📰 Nebius, Eigen AI $643M 인수 — 추론 가속 수직 통합*
5월 1일 공시. Eigen AI는 GPU 추론 컴파일러·커널 전문. Nvidia $2B + Meta $27B에 이은 Nebius의 세 번째 큰 베팅.
🔗 https://www.bloomberg.com/news/articles/2026-05-01/nebius-agrees-to-buy-startup-that-makes-ai-run-faster-cheaper
💡 매트릭스 곱 커널 1줄 = Mistral·Qwen·Llama 토큰 단가 직접 하락.

━━━━━━━━━━━━━━━━━━

🤔 *오늘의 토론 질문*
Mistral 256K 컨텍스트는 추론 시점에서 가장 비싸진다. 한국어 LLM PM이라면 ① Linear Attention ② GPU 커널 최적화 ③ 한국어 형태소 토크나이저 재학습 — 어느 순서로 검증?

💼 *김생근 | AI Human 튜터*
GitHub: https://github.com/kimsanguine/AI_Human
MSG
)
curl -s -X POST "https://api.telegram.org/bot8236282258:AAENnWVQ2wgtJWMzaEuA4tge9qpDL-oXYhg/sendMessage" \
  -d chat_id=8595911950 \
  -d parse_mode=Markdown \
  --data-urlencode "text=$MESSAGE"
