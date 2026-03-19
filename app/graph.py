from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.action import (
    approval_interrupt,
    execute_tool,
    finalize_answer,
    observe_result,
    plan_action,
    route_after_approval,
)
from app.retrieval import (
    answer_check,
    generate_answer,
    grade_docs,
    plan_retrieval,
    retrieve_docs,
    rewrite_query,
    route_retrieval_grade,
)
from app.routing import direct_answer, ingest, route_after_ingest, route_request
from app.state import GraphState


def build_graph() -> StateGraph:
    """LangGraph 워크플로우 정의.

    START
      -> ingest
      -> route
         -> direct      -> END
         -> retrieval   -> (plan -> retrieve -> grade -> [rewrite loop] -> generate -> check) -> END
         -> action      -> (plan -> approval interrupt -> execute -> observe -> finalize) -> END
    """
    graph = StateGraph(GraphState)

    # ===== 공통 노드 =====
    # ingest: 입력 정규화/기본 상태값 세팅
    # route: direct/retrieval/action 중 어떤 워크플로우를 탈지 결정
    # direct: 검색/실행이 필요 없는 간단 응답
    graph.add_node("ingest", ingest)
    graph.add_node("route", route_request)
    graph.add_node("direct", direct_answer)

    # ===== Retrieval 워크플로우 =====
    # plan_retrieval: 검색 질의 생성
    # retrieve_docs: 벡터 스토어 검색
    # grade_docs: 검색 결과 충분성 판단
    # rewrite_query: 질의 재작성(루프)
    # generate_answer/answer_check: 초안 생성 및 최소 검증
    graph.add_node("plan_retrieval", plan_retrieval)
    graph.add_node("retrieve_docs", retrieve_docs)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("answer_check", answer_check)

    # ===== Action 워크플로우 =====
    # plan_action: 실행 계획 수립
    # approval_interrupt: HITL 승인 인터럽트
    # execute_tool: 외부 도구 호출
    # observe_result/finalize_answer: 결과 요약/최종 응답 구성
    graph.add_node("plan_action", plan_action)
    graph.add_node("approval_interrupt", approval_interrupt)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("observe_result", observe_result)
    graph.add_node("finalize_answer", finalize_answer)

    # ===== 기본 진입 엣지 =====
    graph.add_edge(START, "ingest")
    graph.add_edge("ingest", "route")

    # route 노드 출력값(route)에 따라 3개 경로로 분기합니다.
    graph.add_conditional_edges(
        "route",
        route_after_ingest,
        {
            "direct": "direct",
            "retrieval": "plan_retrieval",
            "action": "plan_action",
        },
    )

    graph.add_edge("direct", END)

    # ===== Retrieval 경로 엣지 =====
    graph.add_edge("plan_retrieval", "retrieve_docs")
    graph.add_edge("retrieve_docs", "grade_docs")
    # grade 결과가 insufficient면 rewrite_query로 루프합니다.
    graph.add_conditional_edges(
        "grade_docs",
        route_retrieval_grade,
        {
            "rewrite_query": "rewrite_query",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("rewrite_query", "retrieve_docs")
    graph.add_edge("generate_answer", "answer_check")
    graph.add_edge("answer_check", END)

    # ===== Action 경로 엣지 =====
    graph.add_edge("plan_action", "approval_interrupt")
    # 승인 상태(approved/rejected)에 따라 실행/종료 분기
    graph.add_conditional_edges(
        "approval_interrupt",
        route_after_approval,
        {
            "execute_tool": "execute_tool",
            "finalize_answer": "finalize_answer",
        },
    )
    graph.add_edge("execute_tool", "observe_result")
    graph.add_edge("observe_result", "finalize_answer")
    graph.add_edge("finalize_answer", END)

    return graph


def build_app():
    # MemorySaver는 데모용 인메모리 체크포인터입니다.
    # 프로세스가 내려가면 상태가 사라지므로 운영 환경에서는 외부 영속 저장소 체크포인터를 권장합니다.
    return build_graph().compile(checkpointer=MemorySaver())
