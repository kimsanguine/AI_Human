"""파서 비교 실행기.

    python3 run_compare.py                      # 사용 가능한 파서 전부
    python3 run_compare.py --parser upstage     # 특정 파서만
    python3 run_compare.py --doc L2_요약손익계산서
    python3 run_compare.py --save               # 파서 원본 출력을 out/ 에 저장

출력은 문서×파서 표. 종합 점수 하나로 뭉개지 않고 세 지표를 나란히 보여준다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from parsers import REGISTRY
from score import Score, score_document

HERE = Path(__file__).parent
DATASET = HERE / "dataset"
OUT = HERE / "out"


def load_docs(only: list[str]) -> list[dict]:
    manifest_path = DATASET / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("dataset/ 이 비어 있습니다. 먼저 `make dataset` 를 실행하세요.")
    docs = json.loads(manifest_path.read_text(encoding="utf-8"))
    if only:
        docs = [d for d in docs if d["doc_id"] in only]
    return sorted(docs, key=lambda d: d["level"])


def fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser", action="append", default=[], choices=sorted(REGISTRY))
    ap.add_argument("--doc", action="append", default=[])
    ap.add_argument("--save", action="store_true", help="파서 원본 출력 저장")
    args = ap.parse_args()

    docs = load_docs(args.doc)
    chosen = args.parser or list(REGISTRY)

    active, skipped = [], []
    for name in chosen:
        parser = REGISTRY[name]()
        ok, why = parser.available()
        (active if ok else skipped).append((parser, why))

    for parser, why in skipped:
        print(f"  건너뜀: {parser.name:<14} {why}")
    if skipped:
        print()
    if not active:
        raise SystemExit("실행 가능한 파서가 없습니다.")

    results: list[Score] = []
    for doc in docs:
        truth = json.loads(
            (DATASET / doc["truth"]).read_text(encoding="utf-8"))
        pdf = DATASET / doc["pdf"]
        for parser, _ in active:
            out = parser.run(pdf)
            if args.save:
                OUT.mkdir(exist_ok=True)
                (OUT / f"{doc['doc_id']}__{parser.name}.md").write_text(
                    out.text or f"(오류) {out.error}", encoding="utf-8")
            s = score_document(truth, out.text, parser.name, out.seconds)
            if out.error:
                s.error = out.error
            results.append(s)

    # ---- 출력 ----
    w = max(len(d["doc_id"]) for d in docs) + 2
    print(f"{'문서':<{w}} {'파서':<14} {'텍스트↓':>7} {'표':>6} {'필드':>6} "
          f"{'체크':>6} {'종합':>6} {'초':>6}  비고")
    print("-" * (w + 82))
    for doc in docs:
        for s in [r for r in results if r.doc_id == doc["doc_id"]]:
            note = s.error or s.table_note
            if s.field_misses:
                note += ("  놓친필드: " + ",".join(s.field_misses))
            if s.checkbox_note:
                note += f"  {s.checkbox_note}"
            print(f"{s.doc_id:<{w}} {s.parser:<14} {fmt(s.text_err)} {fmt(s.table)} "
                  f"{fmt(s.fields)} {fmt(s.checkbox)} {fmt(s.overall)} "
                  f"{s.seconds:6.1f}  {note[:60]}")
        print()

    print("파서별 평균 (종합)")
    for parser, _ in active:
        mine = [r for r in results if r.parser == parser.name]
        avg = sum(r.overall for r in mine) / len(mine)
        secs = sum(r.seconds for r in mine)
        print(f"  {parser.name:<14} {parser.kind:<12} {avg * 100:5.1f}점  "
              f"총 {secs:.1f}초")

    if args.save:
        print(f"\n파서 원본 출력: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
