"""PaddleOCR-VL 어댑터 (오픈소스 로컬).

    pip install paddleocr paddlepaddle       # CPU
    pip install paddlepaddle-gpu            # GPU 가 있으면

최초 실행 시 모델을 내려받는다(수 GB). 강의실에서는 미리 받아두거나
강사 이미지에 포함해 두는 편이 낫다.

무료·로컬 실행이라 **문서를 외부로 내보낼 수 없는 금융권 환경**에서
상용 API의 현실적인 대안이 된다. 이 실습의 진짜 질문은
"공짜가 얼마나 따라오는가"이다.
"""

from __future__ import annotations

from pathlib import Path

from .base import ParseResult, Parser


class PaddleOCRVLParser(Parser):
    name = "paddleocr-vl"
    kind = "오픈소스 로컬"

    def __init__(self) -> None:
        self._pipeline = None

    def available(self) -> tuple[bool, str]:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False, "paddleocr 미설치 (pip install paddleocr paddlepaddle)"
        except Exception as exc:   # noqa: BLE001
            # 의존성이 깨진 채로 설치된 경우(예: cryptography 빌드 실패)를
            # 여기서 잡지 않으면 비교 실행 전체가 죽는다.
            return False, f"paddleocr 임포트 실패: {type(exc).__name__}"
        return True, ""

    def _load(self):
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL()
        return self._pipeline

    def parse(self, pdf_path: Path) -> ParseResult:
        pipeline = self._load()
        chunks = []
        for page in pipeline.predict(str(pdf_path)):
            md = getattr(page, "markdown", None)
            if isinstance(md, dict):
                chunks.append(md.get("markdown_texts", ""))
            elif md:
                chunks.append(str(md))
            else:
                chunks.append(str(page))
        return ParseResult(text="\n\n".join(c for c in chunks if c), cost_usd=0.0)
