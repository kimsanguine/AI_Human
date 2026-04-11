# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AI_Human** is a 100-day AI Engineer curriculum repository for non-technical beginners. It serves two primary functions:

1. **Curriculum Storage** — Structured 5-phase curriculum (20 weeks) covering Python basics → ML/DL → Speech AI → LLM → RAG/Agent architecture, tied to real-world product context (PERSO AI).
2. **Automated News System** — A cron-driven daily AI news delivery system that curates 5 articles/day from public sources and commits them to GitHub Mon–Fri at 08:30 KST, with learning connections to the current curriculum chapter.

This is a **content repository** (Markdown + JSON), not a code project.

## Directory Structure

```
/Users/sanguinekim/Documents/AI_Human/
├── curriculum/                           # Curriculum design & configuration
│   ├── mapping.json                       # Central config: chapter schedule, keywords, news tone
│   ├── [설계] RD기반 20주 커리큘럼...md   # 20-week, 5-phase, 800-hour curriculum design
│   └── [설계] 4대 Gap 해소...md           # Gap analysis: LLM fine-tuning, LLMOps, Git, Agent
├── lectures/                              # Lecture materials (교안) per chapter/phase
│   ├── ch01-ch09/                         # Placeholders; intended for 9-subject breakdown
│   ├── phase3-speech-ai-pj1/              # Phase 3 - Speech AI (fully developed)
│   │   ├── 교안_템플릿.md                  # Lecture template for Phase 3+ (채용시장, Ethan's Note, Git checkpoints)
│   │   ├── 마스터인덱스.md                # Day-by-day breakdown W7–W10 (TTS/STT → Whisper → PERSO SDK)
│   │   └── README.md                      # Phase 3 overview, assessment rubric for PJ1
│   └── phase3-llm-pe-lora/                # Phase 3 - LLM/PE/LoRA (3 lecture documents)
├── news/                                  # AI news briefs (automated daily + weekly)
│   ├── daily/                             # Daily briefings: 2026-03-04 to present (weekdays only)
│   └── weekly/                            # Weekly digests: week-01, week-02, etc.
└── scripts/
    └── git_push_aihu.sh                   # Cron automation script (Mon–Fri 08:30 KST)
```

## Key Files & Configuration

### `curriculum/mapping.json`
Central configuration file for the news automation system. Defines 9 chapters (ch01–ch09) with:
- `day_start` / `day_end` — which days of the 100-day course each chapter covers
- `news_keywords` — English search terms for news curation (e.g., "Python", "Machine Learning")
- `news_tone` — split between general AI news and topic-specific news (e.g., "general:3, specific:2")
- `news_count` — always 5 articles per day

Example structure:
```json
{
  "chapters": [
    {
      "id": "ch03",
      "name": "머신러닝과 딥러닝",
      "day_start": 8,
      "day_end": 20,
      "news_keywords": ["machine learning", "deep learning", "neural network"],
      "news_tone": "general:3, specific:2",
      "news_count": 5
    }
  ]
}
```

### `scripts/git_push_aihu.sh`
Cron-scheduled automation script that:
1. Checks if `news/` directory has uncommitted changes
2. Commits changes with message: `daily: YYYY-MM-DD (Day N/100)` or `weekly: week-NN`
3. Pushes to GitHub using SSH key at `~/Documents/.secrets/github_aihu` (private key required)

Run time: **Monday–Friday at 08:30 KST**

If this script fails, check:
- SSH key exists at `~/Documents/.secrets/github_aihu` and has correct permissions (600)
- GitHub deploy key is configured on the repository
- No uncommitted changes outside `news/`

### `lectures/phase3-speech-ai-pj1/교안_템플릿.md`
Standard lecture template for Phase 3 and beyond. All new lecture documents (교안) should follow this structure:

**Required Sections:**
- **채용시장 연결** — How this topic connects to actual hiring market / job requirements
- **Ethan's Note** — Practitioner insight from PM / SaaS background
- **Git Checkpoints** — Code commits / branches to practice with
- **프로젝트 연결** — How this topic ties to the current project milestone (PJ1/PJ2/PJ3)

Example lecture: `phase3-llm-pe-lora/Phase3_W11_LLM기초_PE.md`

## Curriculum Structure (5 Phases, 20 Weeks)

| Phase | Weeks | Topics | Output |
|-------|-------|--------|--------|
| Phase 1 | W1–2 | Python basics, AI tools overview | —— |
| Phase 2 | W3–6 | ML/DL fundamentals, NLP intro | —— |
| Phase 3 | W7–10 | Speech AI (TTS/STT), VITS, Whisper, PERSO SDK | **Project 1** |
| Phase 4 | W11–15 | LLM, Prompt Engineering, LoRA/QLoRA, Streamlit | **Project 2** |
| Phase 5 | W16–20 | LangChain, RAG, LangGraph + MCP + CrewAI | **Project 3** (capstone) |

**Important Note:** Chapters `ch01` through `ch09` are directory placeholders. Actual lecture content is organized by phase in `phase3-*` directories. Future phases will follow the same phase-based structure.

## News Automation Workflow

### Daily Briefing Format (`news/daily/YYYY-MM-DD.md`)

Each daily briefing contains:

1. **Header** — Current day (e.g., "Day 15/100"), current chapter, week number
2. **Five Curated Articles**
   - Source + publish date
   - English title (original)
   - Korean summary (~150–200 words)
   - **학습 연결** — How this article connects to today's curriculum topic
3. **Discussion Question** — Reflection prompt for the cohort

### Weekly Digest Format (`news/weekly/week-NN.md`)

Weekly summaries contain:

1. **Top 3 News of the Week** — Ranked by relevance to curriculum
2. **Key Keyword Summary** — Frequency analysis of trending terms
3. **Day-by-Day Recap Table** — Quick reference for all 5 days
4. **Next Week Preview** — What's coming in the curriculum + relevant news themes

### Content Generation Workflow

The news content is **generated externally** (from OpenClaw vault cron system) and committed to this repo. Files are written to `news/daily/` and `news/weekly/` by an external process, then pushed via `git_push_aihu.sh`.

**Current Status:** Automation active since 2026-03-04. Most recent briefing: 2026-03-24.

## Language Convention

- **Korean (한국어)** — All content, including news summaries, lectures, curriculum docs, and comments
- **English** — JSON keys, shell script logic, code identifiers, technical terminology

## Working with Lectures

When creating or updating lecture documents:

1. **Use the template** — Start from `lectures/phase3-speech-ai-pj1/교안_템플릿.md`
2. **Follow naming convention** — `Phase3_W{week}_{topic}.md` (e.g., `Phase3_W11_LLM기초_PE.md`)
3. **Include all required sections** — 채용시장 연결, Ethan's Note, Git checkpoints, 프로젝트 연결
4. **Link to PERSO AI context** — All projects connect to the same real product
5. **Commit message format** — `lecture: add Phase3_W{week}_{topic}` or `lecture: update Phase3_W{week}_{topic}`

## Working with News Briefs

News briefs are typically generated by the external cron system. If you need to manually add or edit a news brief:

1. **Daily format** — Create `news/daily/YYYY-MM-DD.md`
2. **Follow the template** — Header, 5 articles, learning connections, discussion question
3. **Ensure Korean summaries** — ~150–200 words per article
4. **Link to curriculum chapter** — Reference `curriculum/mapping.json` to find which chapter is active that day
5. **Commit message** — `daily: YYYY-MM-DD (Day N/100)` or `weekly: week-NN`

## Session-End Protocol

When working on curriculum or lecture updates:

1. Verify that changes align with the 5-phase curriculum structure
2. Check that any new lecture follows the template conventions
3. Ensure commit messages follow the format: `lecture: ...`, `curriculum: ...`, `daily: ...`, etc.
4. Note any new content added to memory at `~/.claude/projects/-Users-sanguinekim-Documents-AI-Human/memory/` if it affects future work

## Troubleshooting

**News automation fails to push:**
- Verify SSH key: `ls -la ~/Documents/.secrets/github_aihu` (should be 600)
- Check GitHub deploy key configuration
- Verify no untracked files outside `news/` directory

**Lecture template sections missing:**
- Refer to `lectures/phase3-speech-ai-pj1/교안_템플릿.md` for the canonical structure
- All Phase 3+ lectures must include: 채용시장 연결, Ethan's Note, Git checkpoints, 프로젝트 연결

**Curriculum mapping out of date:**
- Update `curriculum/mapping.json` if chapter day ranges or keywords change
- This is the single source of truth for news keyword selection
