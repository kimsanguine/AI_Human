"""파서 출력 채점기.

세 가지를 따로 잰다. 하나의 종합 점수로 뭉개지 않는 것이 요점이다.
실무에서 중요한 것은 "마크다운이 예쁜가"가 아니라 "필요한 값이 맞는가"이고,
그 둘은 자주 어긋난다.

  1) 텍스트 CER   — 글자를 제대로 읽었는가 (문자 오류율, 낮을수록 좋음)
  2) 표 셀 정확도  — 표의 행/열 구조를 유지했는가
  3) 필드 정확도   — 실무에서 뽑아야 하는 값이 앵커 근처에 정확히 있는가
  4) 체크박스     — ☑/☐ 를 구분했는가 (해당 문서만)

의존성 없음. 순수 파이썬으로 동작한다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field as dc_field

# 앵커에서 정답 값을 찾을 때 허용하는 글자 거리.
# 표 한 행이 대략 이 안에 들어온다.
FIELD_WINDOW = 120


def normalize(s: str) -> str:
    """채점 전 정규화. 파서마다 다른 무해한 차이를 걷어낸다.

    주의: 표 구조를 판단할 때 파이프를 지우면 안 되므로,
    구조를 보존하는 `normalize_layout()` 을 따로 둔다.
    """
    s = normalize_layout(s)
    s = s.replace("|", " ").replace("*", " ").replace("#", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def normalize_layout(s: str) -> str:
    """마크다운 구조 문자(파이프 등)를 살린 채 공백만 정리한다."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[^\S\n]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def compact(s: str) -> str:
    """공백을 전부 제거한 형태. 줄바꿈·정렬 차이를 무시하고 싶을 때."""
    return re.sub(r"\s+", "", normalize(s))


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(truth: str, hyp: str) -> float:
    """문자 오류율. 0.0 이 완벽, 1.0 이면 전부 틀림. (순서에 민감)"""
    t, h = compact(truth), compact(hyp)
    if not t:
        return 0.0 if not h else 1.0
    return min(1.0, levenshtein(t, h) / len(t))


def _best_window_error(needle: str, hay: str) -> float:
    """`needle` 이 `hay` 어딘가에 얼마나 정확히 들어있는지 (0.0 = 완벽)."""
    if not needle:
        return 0.0
    if not hay:
        return 1.0
    if needle in hay:                       # 대부분 여기서 끝난다
        return 0.0
    n = len(needle)
    best = 1.0
    # 길이가 비슷한 구간만 훑는다. 문서가 커도 needle 길이에 비례해서만 늘어난다.
    step = max(1, n // 8)
    for start in range(0, max(1, len(hay) - n + 1), step):
        for width in (n, int(n * 1.25) + 1):
            window = hay[start:start + width]
            if not window:
                continue
            err = levenshtein(needle, window) / n
            if err < best:
                best = err
                if best == 0.0:
                    return 0.0
    return min(1.0, best)


def text_error(truth: str, hyp: str) -> float:
    """줄 순서에 무관한 텍스트 오류율.

    문서 전체를 한 문자열로 놓고 편집거리를 재면, 파서가 본문과 표를
    **다른 순서로** 내놓기만 해도 큰 오류로 잡힌다. 읽기는 완벽한데 배치만
    다른 경우까지 벌점을 주는 셈이라 실제 품질을 왜곡한다.

    그래서 정답을 줄 단위로 쪼갠 뒤, 각 줄이 파서 출력 어딘가에 얼마나
    정확히 들어있는지를 재고 줄 길이로 가중 평균한다.
    """
    lines = [compact(ln) for ln in normalize_layout(truth).splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return 0.0
    hay = compact(hyp)
    if not hay:
        return 1.0
    total = sum(len(ln) for ln in lines)
    weighted = sum(_best_window_error(ln, hay) * len(ln) for ln in lines)
    return min(1.0, weighted / total)


# --------------------------------------------------------------------------
# 표 채점
# --------------------------------------------------------------------------
def extract_md_tables(text: str) -> list[list[list[str]]]:
    """파서 출력에서 마크다운 표를 뽑는다. 없으면 빈 리스트."""
    tables, cur = [], []
    for line in normalize_layout(text).splitlines():
        raw = line.strip()
        if raw.count("|") >= 2 or (cur and raw.count("  ") >= 2 and raw):
            cells = [c.strip() for c in raw.strip("|").split("|")]
            if len(cells) < 2:                       # 파이프가 없는 형태는 공백 분할
                cells = [c for c in re.split(r"\s{2,}", raw) if c]
            if all(re.fullmatch(r"[-: ]+", c) for c in cells):
                continue                             # 마크다운 구분선
            if len(cells) >= 2:
                cur.append(cells)
                continue
        if cur:
            tables.append(cur)
            cur = []
    if cur:
        tables.append(cur)
    return tables


def score_table(truth_table: dict, hyp_text: str) -> tuple[float, str]:
    """정답 표의 각 셀이 파서 출력의 '같은 행'에 있는지 본다.

    행 정렬은 첫 열(과목명/담보명)로 한다. 파서가 행을 통째로 놓치면
    그 행의 모든 셀이 오답이 된다 — 이게 선 없는 표에서 실제로 일어나는 일이다.
    """
    rows = truth_table["rows"]
    total = sum(len(r) for r in rows)
    if total == 0:
        return 1.0, "빈 표"

    hyp_tables = extract_md_tables(hyp_text)
    if not hyp_tables:
        return 0.0, "표를 찾지 못함"

    # 정답 행 수와 가장 가까운 표를 고른다
    best = max(hyp_tables, key=lambda t: -abs(len(t) - len(rows)))
    index = {compact(r[0]): r for r in best if r}

    hit, missing_rows = 0, 0
    for row in rows:
        found = index.get(compact(row[0]))
        if found is None:
            missing_rows += 1
            continue
        found_compact = [compact(c) for c in found]
        for cell in row:
            if compact(cell) in found_compact:
                hit += 1
    note = f"행 {len(rows) - missing_rows}/{len(rows)} 매칭"
    return hit / total, note


# --------------------------------------------------------------------------
# 필드 채점
# --------------------------------------------------------------------------
def score_fields(fields: list[dict], hyp_text: str) -> tuple[float, list[str]]:
    """앵커 주변 FIELD_WINDOW 글자 안에 정답 값이 있는지 본다."""
    if not fields:
        return 1.0, []
    text = compact(hyp_text)
    hit, misses = 0, []
    for f in fields:
        anchor, value = compact(f["anchor"]), compact(f["value"])
        ok = False
        start = 0
        while (pos := text.find(anchor, start)) != -1:
            window = text[pos:pos + FIELD_WINDOW]
            if value in window:
                ok = True
                break
            start = pos + 1
        if ok:
            hit += 1
        else:
            misses.append(f["key"])
    return hit / len(fields), misses


def score_checkbox(expect: dict, hyp_text: str) -> tuple[float, str]:
    """☑ / ☐ 를 구분해서 읽었는가."""
    t = normalize(hyp_text)
    checked = sum(t.count(c) for c in "☑☒✓✔[x][X]")
    unchecked = sum(t.count(c) for c in "☐□")
    want_c, want_u = expect["checked"], expect["unchecked"]
    err = abs(checked - want_c) + abs(unchecked - want_u)
    denom = want_c + want_u
    return max(0.0, 1 - err / denom), f"☑{checked}/{want_c} ☐{unchecked}/{want_u}"


# --------------------------------------------------------------------------
@dataclass
class Score:
    doc_id: str
    parser: str
    text_err: float = 1.0
    table: float = 0.0
    fields: float = 0.0
    checkbox: float | None = None
    table_note: str = ""
    checkbox_note: str = ""
    field_misses: list[str] = dc_field(default_factory=list)
    seconds: float = 0.0
    error: str = ""

    @property
    def overall(self) -> float:
        """참고용 종합 점수. 필드 정확도에 가장 큰 가중치를 둔다."""
        parts = [(1 - self.text_err, 0.25), (self.table, 0.3), (self.fields, 0.45)]
        if self.checkbox is not None:
            parts.append((self.checkbox, 0.15))
        total_w = sum(w for _, w in parts)
        return sum(v * w for v, w in parts) / total_w


def score_document(truth: dict, hyp_text: str, parser: str, seconds: float = 0.0) -> Score:
    s = Score(doc_id=truth["doc_id"], parser=parser, seconds=seconds)
    if not hyp_text.strip():
        s.error = "출력 없음"
        return s
    s.text_err = text_error(truth["text"], hyp_text) if truth.get("text") else 0.0
    if truth["tables"]:
        s.table, s.table_note = score_table(truth["tables"][0], hyp_text)
    s.fields, s.field_misses = score_fields(truth["fields"], hyp_text)
    if truth.get("checkbox"):
        s.checkbox, s.checkbox_note = score_checkbox(truth["checkbox"], hyp_text)
    return s
