from __future__ import annotations

from app.state import GraphState
from app.vector_store import InMemoryVectorStore


# 전역 스토어 1회 초기화:
# - 현재는 데모용 인메모리 스토어를 사용합니다.
# - 실제 서비스에서는 아래 객체를 Chroma/Pinecone/pgvector 래퍼 객체로 교체하면 됩니다.
vector_store = InMemoryVectorStore.bootstrap()


def plan_retrieval(state: GraphState) -> GraphState:
    """사용자 입력을 검색 질의로 변환."""
    return {"retrieval_query": state["normalized_input"]}


def retrieve_docs(state: GraphState) -> GraphState:
    """벡터 스토어에서 관련 문서 조회."""
    docs = vector_store.search(state["retrieval_query"], top_k=3)
    return {"retrieved_docs": docs}


def grade_docs(state: GraphState) -> GraphState:
    """검색 결과가 답변 생성에 충분한지 판정."""
    docs = state.get("retrieved_docs", [])
    if len(docs) >= 1:
        return {"doc_grade": "sufficient"}
    return {"doc_grade": "insufficient"}


def rewrite_query(state: GraphState) -> GraphState:
    """검색 실패 시 질의를 확장해 재시도."""
    cnt = state.get("rewrite_count", 0) + 1
    rewritten = f"{state['retrieval_query']} 환불 정책 안내"
    return {
        "rewrite_count": cnt,
        "retrieval_query": rewritten,
    }


def generate_answer(state: GraphState) -> GraphState:
    """검색 문서 기반으로 답변 초안 생성."""
    docs = state.get("retrieved_docs", [])
    if not docs:
        draft = "관련 문서를 찾지 못했습니다. 추가 정보를 주시면 더 정확히 도와드릴게요."
    else:
        bullets = "\n".join(
            f"- ({d['id']}) {d['title']}: {d['text']}" for d in docs
        )
        draft = f"검색 결과 기반 답변입니다.\n{bullets}"

    return {"answer_draft": draft}


def answer_check(state: GraphState) -> GraphState:
    """초안의 근거 문서 포함 여부를 간단 검증."""
    ok = "- (" in state.get("answer_draft", "")
    if ok:
        return {
            "answer_checked": True,
            "final_answer": state["answer_draft"] + "\n\n(근거 문서 포함)"
        }

    return {
        "answer_checked": False,
        "final_answer": state["answer_draft"] + "\n\n(근거 부족: 재질문 권장)"
    }


def route_retrieval_grade(state: GraphState) -> str:
    """grade_docs 결과에 따라 다음 노드 결정."""
    if state["doc_grade"] == "sufficient":
        return "generate_answer"

    # 최대 2회 재작성 후 종료
    if state.get("rewrite_count", 0) >= 2:
        return "generate_answer"
    return "rewrite_query"
