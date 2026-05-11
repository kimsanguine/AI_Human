#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Telegram 미발송 큐 일괄 처리 스크립트
#
# 동작:
#   1. 새 봇 토큰을 검증 (BotFather에서 발급받은 신규 토큰)
#   2. telegram_pending/*.sh 를 날짜순으로 순차 발송
#   3. 성공 → telegram_sent/ 로 이동 (failed/ 중복 정리)
#      실패 → telegram_failed/ 로 사본 보관
#   4. 401 발생 시 즉시 중단 (토큰 무효)
#   5. 발송 사이 1.5초 sleep (Telegram rate limit 회피)
#
# 사용법:
#   NEW_TOKEN="새:봇토큰" bash ~/Documents/AI_Human/news/send_all_pending.sh
# ─────────────────────────────────────────────────────────────

set -u

OLD_TOKEN="8236282258:AAENnWVQ2wgtJWMzaEuA4tge9qpDL-oXYhg"
NEW_TOKEN="${NEW_TOKEN:-}"

if [ -z "$NEW_TOKEN" ]; then
  echo "✗ NEW_TOKEN 환경변수가 비었습니다."
  echo ""
  echo "  Step 1) Telegram 앱에서 @BotFather 열기"
  echo "  Step 2) /mybots → 봇 선택 → API Token (또는 /revoke 후 재발급)"
  echo "  Step 3) 아래 명령어 실행:"
  echo ""
  echo "    NEW_TOKEN=\"<여기에 새 토큰>\" bash ~/Documents/AI_Human/news/send_all_pending.sh"
  echo ""
  exit 1
fi

BASE="$HOME/Documents/AI_Human/news"
PEND="$BASE/telegram_pending"
SENT="$BASE/telegram_sent"
FAILED="$BASE/telegram_failed"
mkdir -p "$SENT" "$FAILED"

# ── 토큰 검증 ─────────────────────────────────────────────
echo "▶ 새 토큰 검증 중..."
verify=$(curl -s -m 10 "https://api.telegram.org/bot${NEW_TOKEN}/getMe")
if ! echo "$verify" | grep -q '"ok":true'; then
  echo "✗ 새 토큰 검증 실패: $verify"
  exit 1
fi
bot_name=$(echo "$verify" | grep -oE '"username":"[^"]+"' | head -1 | sed 's/"username":"//;s/"//')
echo "✓ 토큰 OK — bot: @${bot_name}"
echo ""

# ── 발송 루프 ─────────────────────────────────────────────
cd "$PEND" 2>/dev/null || { echo "✗ $PEND 없음"; exit 1; }

shopt -s nullglob
files=( $(ls -1 *.sh 2>/dev/null | sort) )
total=${#files[@]}

if [ "$total" -eq 0 ]; then
  echo "✓ 미발송 큐 없음 — 정리 완료 상태"
  exit 0
fi

echo "▶ 미발송 큐: ${total}건"
echo "─────────────────────────────────────────"

success=0
fail=0
idx=0

for f in "${files[@]}"; do
  idx=$((idx + 1))
  printf "[%2d/%2d] %s ... " "$idx" "$total" "$f"

  # 임시 파일에 토큰 치환 후 실행
  tmp=$(mktemp /tmp/tg_send.XXXXXX.sh)
  sed "s|${OLD_TOKEN}|${NEW_TOKEN}|g" "$f" > "$tmp"
  chmod +x "$tmp"
  resp=$(bash "$tmp" 2>&1)
  rm -f "$tmp"

  if echo "$resp" | grep -q '"ok":true'; then
    echo "✓"
    mv "$f" "$SENT/" 2>/dev/null
    rm -f "$FAILED/$f"   # 중복 정리
    success=$((success + 1))
  else
    err=$(echo "$resp" | grep -oE '"description":"[^"]+"' | head -1)
    echo "✗ ${err:-unknown}"
    cp "$f" "$FAILED/"
    fail=$((fail + 1))

    # 401 → 즉시 중단 (토큰 무효)
    if echo "$resp" | grep -q '"error_code":401'; then
      echo ""
      echo "✗ 401 Unauthorized — 토큰이 또 무효입니다. 중단."
      break
    fi

    # 429 → 재시도 가능, 한 번 더 sleep
    if echo "$resp" | grep -q '"error_code":429'; then
      echo "  → rate limit, 30초 대기"
      sleep 30
    fi
  fi

  sleep 1.5  # 일반 rate limit 여유
done

# ── 결과 ─────────────────────────────────────────────────
echo "─────────────────────────────────────────"
echo "  ✓ 성공: $success건 (telegram_sent/ 로 이동)"
echo "  ✗ 실패: $fail건 (telegram_failed/ 사본)"
remaining=$(ls -1 "$PEND"/*.sh 2>/dev/null | wc -l | tr -d ' ')
echo "  ◷ pending 잔여: ${remaining}건"
echo ""

if [ "$success" -gt 0 ]; then
  echo "📂 https://github.com/kimsanguine/AI_Human/tree/main/news/daily"
fi
