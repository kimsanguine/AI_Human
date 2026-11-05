from .base import ParseResult, Parser
from .baseline import BaselineParser
from .paddleocr_vl import PaddleOCRVLParser
from .upstage import UpstageParser

REGISTRY = {
    p.name: p
    for p in (UpstageParser, PaddleOCRVLParser, BaselineParser)
}

__all__ = ["Parser", "ParseResult", "REGISTRY",
           "UpstageParser", "PaddleOCRVLParser", "BaselineParser"]
