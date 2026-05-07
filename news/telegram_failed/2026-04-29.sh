#!/usr/bin/env bash
# Manually send 2026-04-29 daily brief to Telegram (sandbox blocked api.telegram.org)
MESSAGE=$(cat <<'TELEGRAMEOF'
🤖 *AI Human Daily Brief* — 2026-04-29 (수) | Day 41/100
📌 현재 진도: Ch04 자연어 및 음성 데이터

🔥 *TODAY'S TOP AI NEWS*

1️⃣ *DriftSE — 1-step 음성 향상 (arXiv 04-27, Interspeech 2026)*
음성 노이즈 제거를 균형 문제로 정식화해 단 1-step 추론으로 diffusion 수준 품질을 낸 새 생성 프레임워크.
💡 학습 연결: Ch04 음성 표상과 분포 매핑 개념이 generative 음성 모델의 추론 비용 절감으로 이어진 사례.

2️⃣ *Korean-Centric LLM Token Pruning (arXiv 04-22)*
Qwen3·Gemma-3·Llama-3·Aya를 EnKo·EnKoZh 어휘로 압축. 언어 혼동을 줄이고 한국어 번역 성능 향상.
💡 학습 연결: BPE/SentencePiece와 임베딩 테이블이 단순 분절이 아니라 모델 용량 배분 정책 레이어임을 보여줌.

3️⃣ *Do LLM Decoders Listen Fairly? — ASR LLM prior 편향 (arXiv 04-27)*
Speech-aware LLM 기반 ASR이 성별·방언·코드스위칭에 어떤 편향을 주입하는지 정량화.
💡 학습 연결: 어텐션·언어모델링 확률이 음성으로 확장될 때 음향+디코더 LLM 합산 분포의 공정성 평가가 필수.

4️⃣ *Google, Pentagon에 AI 클래시파이드 네트워크 접근 확대 (TechCrunch 04-28)*
Anthropic이 가드레일 요구로 거부한 자리를 OpenAI·xAI에 이어 Google이 차지.

5️⃣ *Sierra, Fragment 인수 — 한 달 내 3건째 통합 인수 (TechCrunch 04-23)*
Bret Taylor의 Sierra가 Opera Tech·Receptive AI에 이어 Fragment까지 인수, 풀스택 에이전트 합병 가속.

🎯 *오늘의 토론 질문*
한국어 LLM 성능을 끌어올릴 때, 데이터 확충 vs 토크나이저·어휘 재설계 — 두 접근의 ROI는 어떤 조건에서 역전되는가?
TELEGRAMEOF
)

curl -s -X POST "https://api.telegram.org/bot8236282258:AAENnWVQ2wgtJWMzaEuA4tge9qpDL-oXYhg/sendMessage" \
  -d chat_id=8595911950 \
  -d parse_mode=Markdown \
  --data-urlencode "text=$MESSAGE"
echo ""
