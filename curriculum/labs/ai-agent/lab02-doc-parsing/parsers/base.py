"""파서 어댑터 인터페이스.

새 파서를 붙이려면 이 파일의 `Parser` 를 상속해 `parse()` 만 구현하면 된다.
수강생 과제로 하나씩 추가하게 하기 좋다 (Mistral OCR, Docling, Gemini 등).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParseResult:
    text: str                 # 마크다운 또는 평문
    seconds: float = 0.0
    error: str = ""
    cost_usd: float | None = None


class Parser:
    name = "base"
    kind = "?"                # "상용 API" / "오픈소스 로컬" / "대조군"

    def available(self) -> tuple[bool, str]:
        """실행 가능한지와 그 이유. (가능여부, 설명)"""
        return True, ""

    def parse(self, pdf_path: Path) -> ParseResult:
        raise NotImplementedError

    def run(self, pdf_path: Path) -> ParseResult:
        ok, why = self.available()
        if not ok:
            return ParseResult(text="", error=why)
        started = time.monotonic()
        try:
            result = self.parse(pdf_path)
        except Exception as exc:                      # noqa: BLE001
            return ParseResult(text="", seconds=time.monotonic() - started,
                               error=f"{type(exc).__name__}: {exc}")
        result.seconds = result.seconds or (time.monotonic() - started)
        return result
