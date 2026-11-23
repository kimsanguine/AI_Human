#!/usr/bin/env bash
# Collects latest daily news + papers from AI_Human and pushes a clipping
# to kimsanguine/llm-brain-private/raw/clippings/YYYY-MM-DD.md
#
# Required env vars:
#   LLMBRAIN_PAT  — GitHub PAT with repo write access to llm-brain-private
#
# Usage: ./scripts/collect_clipping.sh [YYYY-MM-DD]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE="${1:-$(date +%Y-%m-%d)}"
BRAIN_REPO="kimsanguine/llm-brain-private"
BRAIN_CLONE_DIR="$(mktemp -d)"
CLIPPING_PATH="raw/clippings/${DATE}.md"

# ── helpers ──────────────────────────────────────────────────────────────────

log() { echo "[clipping] $*" >&2; }

# 뉴스는 기수마다 폴더가 바뀐다(news/daily -> cohort7/news/daily). 모든 후보에서 파일명 날짜가 가장 최근인 것을 고른다.
# (2026-11-23 수정: 옛 경로만 보던 탓에 6/22 뉴스를 5개월 동안 매일 다시 클리핑했다)
latest_news_file() {
  for f in "$REPO_ROOT"/news/daily/*.md "$REPO_ROOT"/cohort*/news/daily/*.md; do
    [ -f "$f" ] || continue
    name="$(basename "$f" .md)"
    if [[ "$name" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then printf '%s\t%s\n' "$name" "$f"; fi
  done | sort | tail -1 | cut -f2
}

# 두 날짜(YYYY-MM-DD) 사이의 일수
days_between() { python3 -c 'import sys,datetime as d; f=lambda s:d.date.fromisoformat(s); print((f(sys.argv[2])-f(sys.argv[1])).days)' "$1" "$2"; }

# 이보다 오래된 뉴스는 새 클리핑에 넣지 않는다
NEWS_MAX_AGE_DAYS="${NEWS_MAX_AGE_DAYS:-7}"

latest_paper_dirs() {
  # one latest dir per category
  for cat_dir in "$REPO_ROOT/papers"/*/; do
    ls -d "${cat_dir}"[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-* 2>/dev/null | sort | tail -1
  done
}

# ── collect content ───────────────────────────────────────────────────────────

NEWS_FILE="$(latest_news_file)"
NEWS_DATE="없음"
NEWS_REL="없음"
NEWS_BODY="최근 ${NEWS_MAX_AGE_DAYS}일 안에 만들어진 뉴스가 없습니다."
if [ -n "$NEWS_FILE" ]; then
  candidate="$(basename "$NEWS_FILE" .md)"
  age="$(days_between "$candidate" "$DATE")"
  if [ "$age" -ge 0 ] && [ "$age" -le "$NEWS_MAX_AGE_DAYS" ]; then
    NEWS_DATE="$candidate"
    NEWS_REL="${NEWS_FILE#"$REPO_ROOT"/}"
    NEWS_BODY="$(cat "$NEWS_FILE")"
  else
    log "latest news $candidate is ${age} days old (limit ${NEWS_MAX_AGE_DAYS}), skipping news"
  fi
fi
log "news source: $NEWS_DATE ($NEWS_REL)"

# Build papers section
PAPERS_SECTION=""
while IFS= read -r paper_dir; do
  [ -z "$paper_dir" ] && continue
  readme="$paper_dir/README.md"
  [ -f "$readme" ] || continue

  topic="$(grep -m1 "^\*\*Topic\*\*:" "$readme" 2>/dev/null | sed 's/.*: //' || basename "$paper_dir")"
  paper_date="$(grep -m1 "^\*\*Date\*\*:" "$readme" 2>/dev/null | sed 's/.*: //' || echo "unknown")"

  papers_list=""
  while IFS= read -r line; do
    papers_list+="${line}"$'\n'
  done < <(grep -E "^## Paper [0-9]" "$readme" | sed 's/^## /- /')

  arxivs="$(grep "arXiv:" "$readme" | sed 's/.*arXiv: /- /' || true)"

  PAPERS_SECTION+="### ${topic} (${paper_date})\n"
  PAPERS_SECTION+="${papers_list}"
  PAPERS_SECTION+="${arxivs}"
  PAPERS_SECTION+="\n"
done < <(latest_paper_dirs)

# ── build clipping markdown ───────────────────────────────────────────────────

CLIPPING="$(cat <<MARKDOWN
# AI Human Daily Clipping — ${DATE}

> **소스:** [kimsanguine/AI_Human](https://github.com/kimsanguine/AI_Human)
> **뉴스 기준일:** ${NEWS_DATE}
> **수집 시각:** $(date -u +"%Y-%m-%dT%H:%M:%SZ") UTC

---

## 📰 NEWS

${NEWS_BODY}

---

## 📄 PAPERS (최신 카테고리별)

$(echo -e "$PAPERS_SECTION")

---

## 🔗 소스 링크

- 레포: https://github.com/kimsanguine/AI_Human
- 뉴스 원본: \`${NEWS_REL}\`

---
*자동 수집: AI Human Daily Clipping Bot*
MARKDOWN
)"

# ── clone brain repo and push ─────────────────────────────────────────────────

if [ -z "${LLMBRAIN_PAT:-}" ]; then
  log "ERROR: LLMBRAIN_PAT not set. Printing clipping to stdout instead."
  echo "$CLIPPING"
  exit 0
fi

log "cloning $BRAIN_REPO..."
git clone --depth=1 \
  "https://x-access-token:${LLMBRAIN_PAT}@github.com/${BRAIN_REPO}.git" \
  "$BRAIN_CLONE_DIR" 2>/dev/null

mkdir -p "$BRAIN_CLONE_DIR/raw/clippings"
echo "$CLIPPING" > "$BRAIN_CLONE_DIR/$CLIPPING_PATH"

cd "$BRAIN_CLONE_DIR"
git config user.email "bot@ai-human"
git config user.name "AI Human Clipping Bot"
git add "$CLIPPING_PATH"

if git diff --cached --quiet; then
  log "no changes for $DATE, skipping commit."
else
  git commit -m "feat: add ${DATE} daily clipping from ai_human"
  git push origin main
  log "pushed $CLIPPING_PATH to $BRAIN_REPO"
fi

rm -rf "$BRAIN_CLONE_DIR"
