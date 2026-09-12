"""채점기 자체의 검증.

파서를 비교하기 전에 **채점기가 맞는지** 먼저 증명해야 한다.
정답을 그대로 넣으면 만점, 망가뜨린 만큼 점수가 떨어지는지 확인한다.

    python3 test_score.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from score import score_document

DATASET = Path(__file__).parent / "dataset"


def ideal_output(truth: dict) -> str:
    """정답에서 '완벽한 파서 출력'을 만든다. 마크다운 표 + 본문."""
    parts = [truth["text"]]
    for table in truth["tables"]:
        parts.append("| " + " | ".join(table["header"]) + " |")
        parts.append("|" + "---|" * len(table["header"]))
        for row in table["rows"]:
            parts.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(parts)


def drop_rows(text: str, n: int) -> str:
    """표의 마지막 n개 행을 잃어버린 출력."""
    lines = text.splitlines()
    keep = [ln for ln in lines if ln.startswith("|")]
    for ln in keep[-n:]:
        lines.remove(ln)
    return "\n".join(lines)


def typo(text: str, every: int) -> str:
    """every 글자마다 한 글자씩 틀리게 읽은 출력 (OCR 오인식 흉내)."""
    chars = list(text)
    for i in range(0, len(chars), every):
        if chars[i].strip():
            chars[i] = "０"
    return "".join(chars)


CASES = [
    ("완벽한 출력", lambda t, s: s, dict(text_err=(0.0, 0.02), table=(0.99, 1.0), fields=(0.99, 1.0))),
    ("표 2행 누락", lambda t, s: drop_rows(s, 2), dict(table=(0.0, 0.85))),
    ("20자마다 오타", lambda t, s: typo(s, 20), dict(text_err=(0.02, 0.45))),
    ("빈 출력", lambda t, s: "", dict(text_err=(1.0, 1.0), table=(0.0, 0.0), fields=(0.0, 0.0))),
    ("본문만, 표 없음", lambda t, s: t["text"], dict(table=(0.0, 0.0), text_err=(0.0, 0.02))),
]


def main() -> int:
    docs = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    truths = [json.loads((DATASET / d["truth"]).read_text(encoding="utf-8")) for d in docs]

    failures = 0
    print(f"{'케이스':<18} {'문서':<24} {'텍스트↓':>8} {'표':>7} {'필드':>7}  판정")
    print("-" * 78)
    for label, mutate, expect in CASES:
        for truth in truths:
            ideal = ideal_output(truth)
            out = mutate(truth, ideal)
            s = score_document(truth, out, "self-test")

            problems = []
            for key, (lo, hi) in expect.items():
                val = getattr(s, key)
                if not (lo - 1e-9 <= val <= hi + 1e-9):
                    problems.append(f"{key}={val:.3f} ∉ [{lo},{hi}]")
            ok = not problems
            failures += 0 if ok else 1
            print(f"{label:<18} {truth['doc_id']:<24} {s.text_err:8.3f} "
                  f"{s.table:7.3f} {s.fields:7.3f}  "
                  f"{'✅' if ok else '❌ ' + '; '.join(problems)}")
        print()

    if failures:
        print(f"❌ {failures}건이 기대 범위를 벗어났습니다.")
        return 1
    print("✅ 채점기가 기대대로 동작합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
