from __future__ import annotations

import uuid

from langgraph.types import Command

from app.graph import build_app


def run_once(app, text: str, thread_id: str):
    """단일 질의를 실행하는 헬퍼.

    - `thread_id`는 checkpointer(MemorySaver)의 세션 키로 사용됩니다.
    - 같은 `thread_id`를 재사용하면 interrupt 이후에도 상태를 이어서 실행할 수 있습니다.
    """
    config = {"configurable": {"thread_id": thread_id}}
    # LangGraph 실행 입력은 GraphState와 동일한 키 집합을 사용합니다.
    out = app.invoke({"user_input": text, "thread_id": thread_id}, config=config)
    return out


def main() -> None:
    """CLI 엔트리 포인트.

    실행 흐름:
    1) graph.compile()으로 만든 app 객체 생성
    2) thread_id 생성(대화 세션 식별자)
    3) 사용자 입력 1회 수신 후 graph 실행
    4) action 경로에서 interrupt가 나면 승인/거절값으로 resume
    """
    # build_app() 내부에서 StateGraph + MemorySaver(checkpointer)를 구성합니다.
    app = build_app()
    # 매 실행마다 고유 thread_id를 만들지만, 실제 서비스에서는 사용자/세션 단위로 고정하는 것이 일반적입니다.
    thread_id = str(uuid.uuid4())

    print(f"thread_id={thread_id}")
    print("질문을 입력하세요. 예) 환불 정책 알려줘 / 고객에게 환불 처리 메일 보내줘")
    user_input = input("> ").strip()

    # configurable.thread_id가 있어야 interrupt 이후 재개 시 동일 체크포인트를 찾을 수 있습니다.
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_input": user_input, "thread_id": thread_id}, config=config)

    # interrupt 발생 시 "__interrupt__" 키로 중단 페이로드가 반환됩니다.
    if "__interrupt__" in result:
        print("\n[승인 대기]")
        print(result["__interrupt__"])
        decision = input("승인하시겠습니까? (approved/rejected): ").strip().lower()
        # Command(resume=...)로 중단 지점에서 계속 실행합니다.
        resumed = app.invoke(Command(resume=decision), config=config)
        print("\n[최종 응답]")
        print(resumed.get("final_answer", resumed))
        return

    print("\n[최종 응답]")
    print(result.get("final_answer", result))


if __name__ == "__main__":
    main()
