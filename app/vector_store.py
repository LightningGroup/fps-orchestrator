from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InMemoryVectorStore:
    """데모용 의미 검색 저장소(간단 키워드 매칭).

    주의:
    - 실제 '벡터 임베딩' 계산은 하지 않습니다.
    - query 토큰이 문서 문자열(title + text)에 포함되는지로 점수화합니다.
    """

    docs: list[dict[str, str]]

    @classmethod
    def bootstrap(cls) -> "InMemoryVectorStore":
        """데모 문서를 메모리에 로드해 기본 스토어 생성."""
        return cls(
            docs=[
                {
                    "id": "refund-policy-001",
                    "title": "환불 정책",
                    "text": "환불은 결제 후 7일 이내 신청 가능하며, 검토 후 3영업일 내 처리됩니다.",
                },
                {
                    "id": "refund-email-template-002",
                    "title": "환불 안내 메일 템플릿",
                    "text": "고객에게 환불 접수 사실과 예상 처리 일정을 안내하는 템플릿입니다.",
                },
                {
                    "id": "security-approval-003",
                    "title": "실행 승인 정책",
                    "text": "외부 시스템 변경(메일 발송, DB 업데이트)은 승인 후 실행해야 합니다.",
                },
            ]
        )

    def search(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        """질의 문자열과 문서 문자열 간 단순 토큰 매칭 검색.

        Args:
            query: 사용자 질의(또는 재작성 질의)
            top_k: 반환할 최대 문서 수
        """
        q_tokens = set(query.lower().split())

        scored: list[tuple[int, dict[str, str]]] = []
        for d in self.docs:
            # title/text를 하나의 corpus로 보고, 질의 토큰 포함 개수를 score로 사용
            corpus = f"{d['title']} {d['text']}".lower()
            score = sum(1 for t in q_tokens if t in corpus)
            scored.append((score, d))

        # 점수 높은 순 정렬 후, score>0인 문서만 반환
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k] if score > 0]
