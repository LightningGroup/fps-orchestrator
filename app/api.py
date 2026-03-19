from __future__ import annotations

import json
import time
import uuid
import asyncio
from typing import Any, Iterator, Literal

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langgraph.types import Command

from app.graph import build_app


app = FastAPI(title="LangGraph Orchestrator API", version="0.1.0")
_graph_app = build_app()


class StartThreadResponse(BaseModel):
    thread_id: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="사용자 질문/요청")
    thread_id: str | None = Field(
        default=None,
        description="세션 식별자. 없으면 서버가 새로 생성합니다.",
    )


class ChatResponse(BaseModel):
    thread_id: str
    status: Literal["completed", "approval_required"]
    final_answer: str | None = None
    interrupt: dict[str, Any] | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: Literal["approved", "rejected"]


class ApprovalResponse(BaseModel):
    thread_id: str
    status: Literal["completed"]
    final_answer: str


class ChatMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Any
    name: str | None = None
    tool_call_id: str | None = None


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    response_format: dict[str, Any] | None = None
    user: str | None = None


class ResponsesRequest(BaseModel):
    model: str
    input: Any
    instructions: str | None = None
    previous_response_id: str | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    text: dict[str, Any] | None = None
    max_output_tokens: int | None = None
    metadata: dict[str, str] | None = None


def oid(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def error_response(
    status_code: int,
    message: str,
    typ: str = "invalid_request_error",
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": typ,
                "code": code,
            }
        },
    )


def check_auth(authorization: str | None) -> JSONResponse | None:
    if not authorization or not authorization.startswith("Bearer "):
        return error_response(
            401,
            "Missing or invalid Authorization header",
            typ="authentication_error",
        )
    return None


def flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("text", "input_text"):
                out.append(part.get("text", ""))
            elif ptype in ("image_url", "input_image"):
                out.append("[image]")
            elif ptype in ("input_audio", "audio"):
                out.append("[audio]")
            elif ptype in ("file", "input_file"):
                out.append("[file]")
        return "\n".join(x for x in out if x)

    return json.dumps(content, ensure_ascii=False)


def chunk_text(text: str, size: int = 16) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


def sse_event(data: dict[str, Any], event: str | None = None) -> str:
    buf = ""
    if event:
        buf += f"event: {event}\n"
    buf += f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    return buf


def normalize_chat_messages(req: ChatCompletionsRequest) -> str:
    lines: list[str] = []
    for m in req.messages:
        role = "system" if m.role in {"system", "developer"} else m.role
        lines.append(f"{role}: {flatten_content(m.content)}")
    return "\n".join(lines).strip()


def normalize_responses_input(req: ResponsesRequest) -> str:
    chunks: list[str] = []
    if req.instructions:
        chunks.append(f"system: {req.instructions}")

    if isinstance(req.input, str):
        chunks.append(f"user: {req.input}")
    elif isinstance(req.input, list):
        for item in req.input:
            if not isinstance(item, dict):
                chunks.append(f"user: {flatten_content(item)}")
                continue

            item_type = item.get("type")
            role = item.get("role", "user")
            normalized_role = "system" if role in {"developer", "system"} else role

            if item_type == "function_call_output":
                payload = {
                    "call_id": item.get("call_id"),
                    "output": item.get("output"),
                }
                chunks.append(f"tool: {json.dumps(payload, ensure_ascii=False)}")
            else:
                chunks.append(
                    f"{normalized_role}: {flatten_content(item.get('content', item))}"
                )
    else:
        chunks.append(f"user: {json.dumps(req.input, ensure_ascii=False)}")
    return "\n".join(chunks).strip()


def run_gateway_backend(prompt: str) -> str:
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph_app.invoke(
        {"user_input": prompt, "thread_id": thread_id},
        config=config,
    )

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"]
        return f"승인 인터럽트가 필요합니다: {interrupt_payload}"
    return result.get("final_answer", str(result))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/threads", response_model=StartThreadResponse)
def start_thread() -> StartThreadResponse:
    return StartThreadResponse(thread_id=str(uuid.uuid4()))


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph_app.invoke(
        {"user_input": req.message, "thread_id": thread_id},
        config=config,
    )

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"]
        if isinstance(interrupt_payload, tuple) and len(interrupt_payload) > 0:
            value = getattr(interrupt_payload[0], "value", interrupt_payload)
        else:
            value = interrupt_payload

        return ChatResponse(
            thread_id=thread_id,
            status="approval_required",
            interrupt={"detail": value},
        )

    return ChatResponse(
        thread_id=thread_id,
        status="completed",
        final_answer=result.get("final_answer", str(result)),
    )


@app.post("/chat/approval", response_model=ApprovalResponse)
def approve_chat(req: ApprovalRequest) -> ApprovalResponse:
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        resumed = _graph_app.invoke(Command(resume=req.decision), config=config)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="유효한 인터럽트 상태를 찾지 못했습니다. thread_id를 확인하세요.",
        ) from exc

    return ApprovalResponse(
        thread_id=req.thread_id,
        status="completed",
        final_answer=resumed.get("final_answer", str(resumed)),
    )


@app.get("/v1/models")
def list_models(authorization: str | None = Header(default=None)):
    auth_error = check_auth(authorization)
    if auth_error:
        return auth_error

    return {
        "object": "list",
        "data": [
            {
                "id": "corp-gpt",
                "object": "model",
                "created": 0,
                "owned_by": "corp",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def create_chat_completion(
    req: ChatCompletionsRequest,
    authorization: str | None = Header(default=None),
):
    auth_error = check_auth(authorization)
    if auth_error:
        return auth_error

    request_id = oid("req_")
    started = time.time()
    created = int(time.time())
    completion_id = oid("chatcmpl-")
    headers = {
        "x-request-id": request_id,
        "openai-processing-ms": str(int((time.time() - started) * 1000)),
    }

    try:
        prompt = normalize_chat_messages(req)
        answer_text = run_gateway_backend(prompt)
    except Exception as exc:
        return error_response(502, f"Upstream LLM error: {exc}", typ="api_error")

    if req.stream:

        async def event_gen():
            first = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

            for piece in chunk_text(answer_text):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": piece},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            last = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers=headers,
        )

    body = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return JSONResponse(content=body, headers=headers)


@app.post("/v1/responses")
async def create_response(
    req: ResponsesRequest,
    authorization: str | None = Header(default=None),
):
    auth_error = check_auth(authorization)
    if auth_error:
        return auth_error

    request_id = oid("req_")
    started = time.time()
    created = int(time.time())
    response_id = oid("resp_")
    message_id = oid("msg_")
    headers = {
        "x-request-id": request_id,
        "openai-processing-ms": str(int((time.time() - started) * 1000)),
    }

    try:
        prompt = normalize_responses_input(req)
        answer_text = run_gateway_backend(prompt)
    except Exception as exc:
        return error_response(502, f"Upstream LLM error: {exc}", typ="api_error")

    final_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": req.instructions,
        "model": req.model,
        "output": [
            {
                "id": message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": answer_text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": answer_text,
        "previous_response_id": req.previous_response_id,
        "tools": req.tools or [],
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }

    if req.stream:

        async def event_gen():
            yield sse_event(
                {
                    "type": "response.created",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created,
                        "status": "in_progress",
                        "model": req.model,
                        "output": [],
                    },
                },
                event="response.created",
            )

            for piece in chunk_text(answer_text):
                yield sse_event(
                    {
                        "type": "response.output_text.delta",
                        "item_id": message_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": piece,
                    },
                    event="response.output_text.delta",
                )
                await asyncio.sleep(0)

            yield sse_event(
                {
                    "type": "response.output_text.done",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": answer_text,
                },
                event="response.output_text.done",
            )

            yield sse_event(
                {
                    "type": "response.completed",
                    "response": final_response,
                },
                event="response.completed",
            )

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers=headers,
        )

    return JSONResponse(content=final_response, headers=headers)
