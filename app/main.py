from __future__ import annotations

"""LangGraph 앱 실행을 위한 공용 헬퍼 모듈.

이 모듈은 과거에 `python -m app.main` 형태의 CLI 데모 진입점 역할도 함께 수행했습니다.
요청사항에 따라 **표준 입력(input) 기반 CLI 실행 코드는 제거**하고,
서버/API 계층에서 재사용하기 쉬운 함수형 인터페이스만 남겼습니다.

핵심 목적:
- LangGraph app 인스턴스 생성
- 단일 사용자 요청 실행
- interrupt 이후 승인/거절 값으로 재개
"""

import uuid
from typing import Any

from langgraph.types import Command

from app.graph import build_app


def create_app():
    """LangGraph 애플리케이션 인스턴스를 생성합니다.

    상세 설명:
    - `build_app()` 내부에서 `StateGraph`를 조립하고,
      `MemorySaver` 체크포인터를 연결한 실행 가능 객체를 반환합니다.
    - 호출 시점마다 새 app 객체를 만들 수 있으며,
      서버 프로세스에서는 보통 모듈 전역으로 1회 생성 후 재사용합니다.

    Returns:
        graph.compile() 결과로 만들어진 실행 가능한 LangGraph app 객체.
    """
    return build_app()


def new_thread_id() -> str:
    """새 대화 세션 식별자(thread_id)를 생성합니다.

    `thread_id`는 체크포인터의 세션 키로 사용되며,
    같은 값을 재사용해야 interrupt 이후에도 동일한 실행 상태를 이어갈 수 있습니다.
    """
    return str(uuid.uuid4())


def run_once(app: Any, text: str, thread_id: str) -> dict[str, Any]:
    """단일 질의를 실행합니다.

    Args:
        app: `create_app()` 혹은 `build_app()`으로 생성한 LangGraph app.
        text: 사용자 입력 텍스트.
        thread_id: 세션 식별자.

    Returns:
        LangGraph 노드 실행 결과 딕셔너리.
        - 일반 완료 시 `final_answer` 등을 포함할 수 있습니다.
        - 승인 인터럽트 발생 시 `__interrupt__` 키를 포함합니다.

    동작 포인트:
    - `configurable.thread_id`를 config로 전달해 체크포인터에서
      현재 세션 상태를 조회/갱신할 수 있게 합니다.
    - 입력 payload에도 `thread_id`를 넣어 노드 레벨 로직에서
      동일 식별자를 참조할 수 있도록 유지합니다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({"user_input": text, "thread_id": thread_id}, config=config)


def resume_once(app: Any, thread_id: str, decision: str) -> dict[str, Any]:
    """interrupt 지점에서 승인/거절 값을 넣어 실행을 재개합니다.

    Args:
        app: LangGraph app 객체.
        thread_id: 재개할 세션 식별자.
        decision: 승인 결과 문자열(예: ``approved`` / ``rejected``).

    Returns:
        재개 이후 LangGraph 실행 결과 딕셔너리.

    구현 메모:
    - LangGraph는 `Command(resume=...)`를 사용해
      마지막 interrupt 지점으로 값을 주입하며 실행을 이어갑니다.
    - 반드시 처음 interrupt를 만들었던 것과 **같은 thread_id**를 전달해야 합니다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke(Command(resume=decision), config=config)
