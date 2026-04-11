#!/bin/bash
# AI Human 자동 GitHub Push 스크립트
# 매일 오전 8:30 (월~금) macOS cron으로 실행

REPO="$HOME/Documents/AI_Human"
LOG="$HOME/Documents/AI_Human/scripts/push.log"
SSH_KEY="$HOME/Documents/.secrets/github_aihu"

cd "$REPO" || exit 1

# lock 파일 잔재 제거
[ -f .git/index.lock ] && rm -f .git/index.lock

# 변경사항 있을 때만 커밋 + push
if [[ -n $(git status --porcelain news/) ]]; then
  git add news/
  git commit -m "auto: $(date '+%Y-%m-%d') daily push"
  GIT_SSH_COMMAND="ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    git push origin main >> "$LOG" 2>&1
  echo "$(date '+%Y-%m-%d %H:%M') ✅ push 완료" >> "$LOG"
else
  echo "$(date '+%Y-%m-%d %H:%M') — 변경사항 없음" >> "$LOG"
fi
