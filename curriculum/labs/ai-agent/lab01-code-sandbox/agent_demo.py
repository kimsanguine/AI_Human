"""샌드박스를 '도구'로 붙인 최소 에이전트.

Lab 1의 마지막 단계. 지금까지 만든 run_python 을 LLM 이 호출할 수 있는
도구로 등록하면, 에이전트는 스스로 분석 코드를 짜고 → 격리 실행하고 →
오류가 나면 고쳐서 다시 실행한다.

실행:
    pip install anthropic
    export ANTHROPIC_API_KEY=...        # 또는 `ant auth login`
    python agent_demo.py "부채비율이 가장 높은 3개 회사와 그 수치를 알려줘"
"""

from __future__ import annotations

import sys
from pathlib import Path

import anthropic
from anthropic import beta_tool

from sandbox import run_python

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / ".out" / "agent"

SYSTEM = """당신은 금융 데이터 분석 어시스턴트입니다.

분석은 반드시 run_python 도구로 수행하세요. 머릿속으로 숫자를 계산하지 마세요.
- 데이터는 컨테이너의 /work/in/financials.csv 에 읽기 전용으로 있습니다.
- 그림·표 파일이 필요하면 /work/out/ 아래에 저장하세요.
- 샌드박스는 외부 네트워크가 차단돼 있습니다. 데이터를 내려받으려 하지 마세요.
- 코드가 실패하면 오류 메시지를 읽고 수정해서 다시 실행하세요.

최종 답변에는 숫자의 근거(어떤 계산식을 썼는지)를 한 줄로 덧붙이세요.
이 데이터는 실습용 가상 데이터이며, 투자 권유가 아님을 명시하세요."""


@beta_tool
def run_python_in_sandbox(code: str) -> str:
    """격리된 컨테이너에서 파이썬 코드를 실행하고 stdout 을 돌려준다.

    pandas, numpy, matplotlib 를 쓸 수 있다. 네트워크는 차단돼 있고,
    /work/in 은 읽기 전용, /work/out 만 쓰기 가능하다.

    Args:
        code: 실행할 파이썬 소스 전체. 결과는 print 로 출력할 것.
    """
    result = run_python(code, data_dir=DATA, out_dir=OUT)
    print(f"  [sandbox] {'ok' if result.ok else 'fail'} "
          f"({result.duration:.1f}s, {len(code)}자)", file=sys.stderr)
    return result.summary()


def main() -> None:
    question = " ".join(sys.argv[1:]) or "부채비율 상위 3개 회사를 알려줘."
    client = anthropic.Anthropic()

    runner = client.beta.messages.tool_runner(
        model="claude-opus-5",
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        tools=[run_python_in_sandbox],
        messages=[{"role": "user", "content": question}],
    )

    for message in runner:
        for block in message.content:
            if block.type == "text" and block.text.strip():
                print(block.text)

    if OUT.exists():
        files = sorted(p.name for p in OUT.iterdir() if p.is_file())
        if files:
            print(f"\n생성된 파일: {', '.join(files)}  ({OUT})")


if __name__ == "__main__":
    main()
