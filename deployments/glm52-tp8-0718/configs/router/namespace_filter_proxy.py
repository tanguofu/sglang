#!/usr/bin/env python3
"""
Tool filter + Anthropic-to-OpenAI proxy for SGLang Rust router.

Two responsibilities:
1. Strip unsupported tool types (namespace, web_search, etc.) from
   /v1/chat/completions and /v1/responses requests before forwarding
   to the Rust router (which only accepts 4 tool types).

2. Convert /v1/messages (Anthropic Messages API) to /v1/chat/completions
   (OpenAI Chat Completion API) because:
   a. The Rust router returns 404 for /v1/messages
   b. SGLang's native /v1/messages endpoint has a segfault bug

The Rust router runs on ROUTER_PORT (30081), this proxy listens on
PROXY_PORT (30080).
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid

import aiohttp
from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tool-filter-proxy")

PROXY_PORT = int(os.environ.get("PROXY_PORT", "30080"))
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "30081"))
ROUTER_URL = f"http://127.0.0.1:{ROUTER_PORT}"

MAX_BODY = 512 * 1024 * 1024

SUPPORTED_TOOL_TYPES = frozenset({
    "function",
})


# ---------------------------------------------------------------------------
# Anthropic → OpenAI request conversion
# ---------------------------------------------------------------------------

def anthropic_to_openai(data: dict) -> dict:
    """Convert Anthropic Messages API request to OpenAI Chat Completion request."""
    messages = []

    # System prompt: Anthropic uses top-level "system" field
    system = data.get("system")
    if system:
        if isinstance(system, list):
            # Array of content blocks
            system_text = " ".join(
                block.get("text", "") for block in system
                if isinstance(block, dict) and block.get("type") == "text"
            )
        else:
            system_text = str(system)
        if system_text.strip():
            messages.append({"role": "system", "content": system_text})

    # Messages: Anthropic content blocks → OpenAI content string
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            tool_results = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    # This shouldn't appear in user messages normally,
                    # but handle it as a function call
                    pass
                elif btype == "tool_result":
                    tool_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    if isinstance(result_content, list):
                        result_text = " ".join(
                            b.get("text", "") for b in result_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        result_text = str(result_content)
                    tool_results.append({"tool_id": tool_id, "result": result_text})

            if tool_results:
                # Convert tool results to OpenAI tool messages
                for tr in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_id"],
                        "content": tr["result"],
                    })
            if text_parts:
                text_content = "\n".join(text_parts)
                if role == "assistant":
                    messages.append({"role": "assistant", "content": text_content})
                else:
                    messages.append({"role": role, "content": text_content})
        elif content is None:
            # Assistant message with tool_calls only
            if role == "assistant":
                messages.append({"role": "assistant", "content": ""})

    # Build OpenAI request
    openai_req = {
        "model": data.get("model", ""),
        "messages": messages,
        "max_tokens": data.get("max_tokens", 4096),
    }

    # Optional fields
    if "temperature" in data:
        openai_req["temperature"] = data["temperature"]
    if "top_p" in data:
        openai_req["top_p"] = data["top_p"]
    if "stream" in data:
        openai_req["stream"] = data["stream"]
    if "stop" in data:
        openai_req["stop"] = data["stop"]

    # Convert tools
    anthropic_tools = data.get("tools", [])
    if anthropic_tools:
        openai_tools = []
        for tool in anthropic_tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function":
                # Already OpenAI format
                openai_tools.append(tool)
            elif "name" in tool and "input_schema" in tool:
                # Anthropic tool format → OpenAI function format
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    },
                })
        if openai_tools:
            openai_req["tools"] = openai_tools

    # Convert tool_choice
    tc = data.get("tool_choice")
    if tc:
        if isinstance(tc, dict):
            if tc.get("type") == "auto":
                openai_req["tool_choice"] = "auto"
            elif tc.get("type") == "any":
                openai_req["tool_choice"] = "required"
            elif tc.get("type") == "tool" and "name" in tc:
                openai_req["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }
        elif isinstance(tc, str):
            openai_req["tool_choice"] = tc

    return openai_req


# ---------------------------------------------------------------------------
# OpenAI → Anthropic response conversion (non-streaming)
# ---------------------------------------------------------------------------

def openai_to_anthropic(openai_resp: dict, model: str) -> dict:
    """Convert OpenAI Chat Completion response to Anthropic Messages response."""
    choices = openai_resp.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    content = []
    # Handle reasoning_content (GLM-5.2 thinking)
    reasoning = message.get("reasoning_content")
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})

    # Handle text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Handle tool_calls
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
                "input": args,
            })

    # Map finish_reason
    finish = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish, "end_turn")

    usage = openai_resp.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    reasoning_tokens = usage.get("reasoning_tokens", 0)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# OpenAI Responses API → Chat Completions conversion
# ---------------------------------------------------------------------------

def responses_to_chat(data: dict) -> dict:
    """Convert OpenAI Responses API request to Chat Completions request.

    Responses API uses `input` (string or array of input items) instead of
    `messages`. Input items can be:
      - {type: "message", role, content}
      - {type: "function_call", name, arguments, call_id}
      - {type: "function_call_output", call_id, output}
    """
    messages = []
    inp = data.get("input", [])

    # System prompt
    instructions = data.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "")
            if itype == "message":
                role = item.get("role", "user")
                # Map "developer" role (Responses API) to "system" (Chat Completions)
                if role == "developer":
                    role = "system"
                content = item.get("content", "")
                if isinstance(content, list):
                    # Extract text from content blocks — Responses API uses
                    # "input_text" for user messages and "output_text" for
                    # assistant messages, while Chat Completions uses "text".
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "")
                            if btype in ("text", "input_text", "output_text"):
                                text_parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = "\n".join(text_parts)
                # sglang rejects empty content — use space as placeholder
                if not content:
                    content = " "
                messages.append({"role": role, "content": content})
            elif itype == "function_call":
                # Assistant tool call — use null content (sglang rejects empty string)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id", f"call_{uuid.uuid4().hex[:12]}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
            elif itype == "function_call_output":
                # Tool result — ensure non-empty content
                output = item.get("output", "")
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": output if output else " ",
                })
            elif itype == "reasoning":
                # Skip reasoning items — they'll be regenerated
                pass

    # Build chat completions request
    chat_req = {
        "model": data.get("model", ""),
        "messages": messages,
        "max_tokens": data.get("max_output_tokens", 4096),
    }

    # Optional fields
    if "temperature" in data:
        chat_req["temperature"] = data["temperature"]
    if "top_p" in data:
        chat_req["top_p"] = data["top_p"]
    if "stream" in data:
        chat_req["stream"] = data["stream"]
    if "reasoning" in data:
        reason = data["reasoning"]
        if isinstance(reason, dict) and "effort" in reason:
            chat_req["reasoning_effort"] = reason["effort"]

    # Convert tools: Responses API tools have different format
    resp_tools = data.get("tools", [])
    if resp_tools:
        openai_tools = []
        for tool in resp_tools:
            if not isinstance(tool, dict):
                continue
            ttype = tool.get("type", "")
            if ttype == "function" and "function" in tool:
                # Already in OpenAI chat completions format
                openai_tools.append(tool)
            elif ttype == "function" or ttype == "function_definition" or "name" in tool:
                # Responses API function definition — convert to chat completions format.
                # Codex /v1/responses sends {type: "function", name, parameters, ...}
                # but chat completions expects {type: "function", function: {name, ...}}.
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", tool.get("input_schema", {})),
                    },
                })
        if openai_tools:
            chat_req["tools"] = openai_tools

    # Convert tool_choice
    tc = data.get("tool_choice")
    if tc:
        if isinstance(tc, str):
            chat_req["tool_choice"] = tc
        elif isinstance(tc, dict):
            if tc.get("type") == "auto":
                chat_req["tool_choice"] = "auto"
            elif tc.get("type") == "required":
                chat_req["tool_choice"] = "required"
            elif tc.get("type") == "function" and "name" in tc:
                chat_req["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc["name"]},
                }

    return chat_req


def chat_to_responses(chat_resp: dict, model: str) -> dict:
    """Convert Chat Completions response to Responses API response."""
    choices = chat_resp.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    output = []
    # Reasoning
    reasoning = message.get("reasoning_content")
    if reasoning:
        output.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": reasoning}]})

    # Text content
    text = message.get("content")
    if text:
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })

    # Tool calls
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        if isinstance(tc, dict):
            func = tc.get("function", {})
            output.append({
                "type": "function_call",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                "name": func.get("name", ""),
                "arguments": func.get("arguments", "{}"),
            })

    # Map finish_reason
    finish = choice.get("finish_reason", "stop")
    status_map = {
        "stop": "completed",
        "length": "incomplete",
        "tool_calls": "completed",
        "function_call": "completed",
    }
    status = status_map.get(finish, "completed")

    usage = chat_resp.get("usage", {})

    return {
        "id": f"resp_{uuid.uuid4().hex[:24]}",
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


class ResponsesStreamConverter:
    """Convert OpenAI Chat Completions SSE stream to Responses API SSE stream."""

    def __init__(self, model: str):
        self.model = model
        self.response_id = f"resp_{uuid.uuid4().hex[:24]}"
        self._created = False
        self._text_started = False
        self._text_full = ""
        self._tool_indices = {}   # tc_idx → output_index (seq)
        self._tool_seen = set()
        self._tool_args = {}      # tc_idx → accumulated arguments string
        self._tool_ids = {}       # tc_idx → tool call id (e.g. "call_abc123")
        self._tool_names = {}     # tc_idx → function name (e.g. "shell")
        self._seq = 0
        self._reasoning_items = []
        self._text_seq = 0

    def _sse(self, event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    def convert_chunk(self, chunk_data: dict) -> str:
        output = ""

        if not self._created:
            self._created = True
            output += self._sse("response.created", {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.model,
                    "output": [],
                },
            })
            output += self._sse("response.in_progress", {
                "type": "response.in_progress",
                "response": {"id": self.response_id, "status": "in_progress"},
            })

        choices = chunk_data.get("choices", [])
        if not choices:
            return output

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Reasoning — accumulate into a single reasoning item
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not self._reasoning_items:
                seq = self._next_seq()
                self._reasoning_items.append({"seq": seq, "text": ""})
                output += self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": seq,
                    "item": {"type": "reasoning", "id": f"rs_{uuid.uuid4().hex[:12]}"},
                })
            ri = self._reasoning_items[-1]
            ri["text"] += reasoning
            output += self._sse("response.reasoning_text.delta", {
                "type": "response.reasoning_text.delta",
                "output_index": ri["seq"],
                "delta": reasoning,
            })

        # Text content
        text = delta.get("content")
        if text:
            if not self._text_started:
                self._text_started = True
                seq = self._next_seq()
                self._text_seq = seq
                output += self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": seq,
                    "item": {"type": "message", "role": "assistant", "content": []},
                })
                output += self._sse("response.content_part.added", {
                    "type": "response.content_part.added",
                    "output_index": seq,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": ""},
                })
            self._text_full += text
            output += self._sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "output_index": self._text_seq,
                "content_index": 0,
                "delta": text,
            })

        # Tool calls
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            tc_idx = tc.get("index", 0)
            tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:12]}")
            tc_name = func.get("name", "")
            tc_args = func.get("arguments", "")

            if tc_idx not in self._tool_seen:
                self._tool_seen.add(tc_idx)
                seq = self._next_seq()
                self._tool_indices[tc_idx] = seq
                self._tool_args[tc_idx] = ""
                self._tool_ids[tc_idx] = tc_id
                self._tool_names[tc_idx] = tc_name
                output += self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": seq,
                    "item": {
                        "type": "function_call",
                        "id": tc_id,
                        "call_id": tc_id,
                        "name": tc_name,
                        "arguments": "",
                    },
                })

            if tc_args:
                self._tool_args[tc_idx] = (self._tool_args.get(tc_idx, "")) + tc_args
                seq = self._tool_indices.get(tc_idx, 0)
                output += self._sse("response.function_call_arguments.delta", {
                    "type": "response.function_call_arguments.delta",
                    "output_index": seq,
                    "delta": tc_args,
                })

        # Finish — emit done events for all open items, then completed
        if finish_reason:
            # Close reasoning items
            for ri in self._reasoning_items:
                output += self._sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": ri["seq"],
                    "item": {
                        "type": "reasoning",
                        "id": f"rs_{uuid.uuid4().hex[:12]}",
                        "summary": [{"type": "summary_text", "text": ri["text"]}],
                    },
                })

            # Close text item
            if self._text_started:
                output += self._sse("response.output_text.done", {
                    "type": "response.output_text.done",
                    "output_index": self._text_seq,
                    "content_index": 0,
                    "text": self._text_full,
                })
                output += self._sse("response.content_part.done", {
                    "type": "response.content_part.done",
                    "output_index": self._text_seq,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self._text_full},
                })
                output += self._sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": self._text_seq,
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": self._text_full}],
                    },
                })

            # Close tool call items
            for tc_idx, seq in self._tool_indices.items():
                args = self._tool_args.get(tc_idx, "")
                tc_id = self._tool_ids.get(tc_idx, f"call_{tc_idx}")
                tc_name = self._tool_names.get(tc_idx, "")
                output += self._sse("response.output_item.done", {
                    "type": "response.output_item.done",
                    "output_index": seq,
                    "item": {
                        "type": "function_call",
                        "id": tc_id,
                        "call_id": tc_id,
                        "name": tc_name,
                        "arguments": args,
                    },
                })

            usage = chunk_data.get("usage", {}) or {}
            output += self._sse("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "completed",
                    "model": self.model,
                    "output": [],
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                },
            })

        return output

    def finalize(self) -> str:
        if not self._created:
            self._created = True
            return self._sse("response.created", {
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "object": "response",
                    "status": "completed",
                    "model": self.model,
                    "output": [],
                },
            }) + self._sse("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": self.response_id,
                    "status": "completed",
                    "model": self.model,
                    "output": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            })
        return ""


# ---------------------------------------------------------------------------
# OpenAI → Anthropic SSE streaming conversion
# ---------------------------------------------------------------------------

class AnthropicStreamConverter:
    """Convert OpenAI SSE stream to Anthropic SSE stream."""

    def __init__(self, model: str):
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.content_index = 0
        self.current_block_type = None  # "thinking" or "text" or "tool_use"
        self.block_started = False
        self.output_tokens = 0
        self.input_tokens = 0
        self._message_started = False
        self._first_chunk = True
        self._tool_block_indices = {}  # tc_index → content_block_index
        self._tool_seen = set()  # tc_index values that have started a block

    def _sse(self, event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def _start_message(self) -> str:
        self._message_started = True
        return self._sse("message_start", {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.input_tokens,
                    "output_tokens": 0,
                },
            },
        })

    def _start_block(self, block_type: str, **extra) -> str:
        self.current_block_type = block_type
        self.block_started = True
        block = {"type": block_type}
        if block_type == "text":
            block["text"] = ""
        elif block_type == "thinking":
            block["thinking"] = ""
        elif block_type == "tool_use":
            block.update(extra)
        return self._sse("content_block_start", {
            "type": "content_block_start",
            "index": self.content_index,
            "content_block": block,
        })

    def _stop_block(self) -> str:
        event = self._sse("content_block_stop", {
            "type": "content_block_stop",
            "index": self.content_index,
        })
        self.content_index += 1
        self.current_block_type = None
        self.block_started = False
        return event

    def _delta(self, delta_type: str, text: str, index: int = None) -> str:
        """Emit a content_block_delta with the correct field name per delta type.

        Anthropic SSE spec:
          - text_delta:       {"type": "text_delta", "text": "..."}
          - thinking_delta:   {"type": "thinking_delta", "thinking": "..."}
          - input_json_delta: {"type": "input_json_delta", "partial_json": "..."}
        """
        if index is None:
            index = self.content_index
        field_map = {
            "text_delta": "text",
            "thinking_delta": "thinking",
            "input_json_delta": "partial_json",
        }
        field = field_map.get(delta_type, "text")
        return self._sse("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": delta_type, field: text},
        })

    def convert_chunk(self, chunk_data: dict) -> str:
        """Convert one OpenAI SSE data chunk to Anthropic SSE events."""
        output = ""

        if self._first_chunk:
            output += self._start_message()
            self._first_chunk = False

        choices = chunk_data.get("choices", [])
        if not choices:
            return output

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle reasoning_content (thinking)
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if self.current_block_type != "thinking":
                if self.block_started:
                    output += self._stop_block()
                output += self._start_block("thinking")
            output += self._delta("thinking_delta", reasoning)
            self.output_tokens += 1

        # Handle regular text content
        text = delta.get("content")
        if text:
            if self.current_block_type != "text":
                if self.block_started:
                    output += self._stop_block()
                output += self._start_block("text")
            output += self._delta("text_delta", text)
            self.output_tokens += 1

        # Handle tool_calls
        tool_calls = delta.get("tool_calls") or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function", {})
            tc_idx = tc.get("index", 0)
            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
            tc_name = func.get("name", "")
            tc_args = func.get("arguments", "")

            if tc_idx not in self._tool_seen:
                # New tool call — stop previous block, start a tool_use block
                if self.block_started:
                    output += self._stop_block()
                self._tool_block_indices[tc_idx] = self.content_index
                self._tool_seen.add(tc_idx)
                output += self._start_block(
                    "tool_use",
                    id=tc_id,
                    name=tc_name,
                    input={},
                )

            # Stream tool arguments as input_json_delta (Anthropic spec)
            if tc_args:
                block_idx = self._tool_block_indices.get(tc_idx, self.content_index)
                output += self._delta("input_json_delta", tc_args, index=block_idx)
                self.output_tokens += 1

        # Handle finish
        if finish_reason:
            if self.block_started:
                output += self._stop_block()

            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "function_call": "tool_use",
            }
            stop_reason = stop_reason_map.get(finish_reason, "end_turn")

            usage = chunk_data.get("usage", {})
            if usage:
                self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
                self.output_tokens = usage.get("completion_tokens", self.output_tokens)

            output += self._sse("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason,
                    "stop_sequence": None,
                },
                "usage": {
                    "output_tokens": self.output_tokens,
                },
            })
            output += self._sse("message_stop", {"type": "message_stop"})

        return output

    def finalize(self) -> str:
        """Ensure stream is properly terminated."""
        output = ""
        if self._first_chunk:
            # No data received at all
            output += self._start_message()
            output += self._start_block("text")
            output += self._delta("text_delta", "")
            output += self._stop_block()
            output += self._sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            })
            output += self._sse("message_stop", {"type": "message_stop"})
        elif self.block_started:
            output += self._stop_block()
            output += self._sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            })
            output += self._sse("message_stop", {"type": "message_stop"})
        return output


# ---------------------------------------------------------------------------
# Request handlers
# ---------------------------------------------------------------------------

def filter_tools(data: dict) -> tuple[dict, str]:
    """Strip unsupported tool types and unwrap namespace tools.

    Returns (modified_data, log_detail).
    """
    if not isinstance(data, dict) or not isinstance(data.get("tools"), list):
        return data, ""

    original = len(data["tools"])
    kept = []
    removed_types = {}
    stripped_names = set()

    for t in data["tools"]:
        if not isinstance(t, dict):
            kept.append(t)
            continue

        ttype = t.get("type", "")

        if ttype == "namespace":
            # Unwrap namespace tools: extract inner function tools
            inner = t.get("tools", [])
            if isinstance(inner, list):
                for inner_t in inner:
                    if isinstance(inner_t, dict) and inner_t.get("type") == "function":
                        kept.append(inner_t)
            removed_types["namespace"] = removed_types.get("namespace", 0) + 1
        elif ttype in SUPPORTED_TOOL_TYPES:
            kept.append(t)
        else:
            removed_types[ttype or "(none)"] = removed_types.get(ttype or "(none)", 0) + 1
            # Track stripped function names for tool_choice sanitization
            if ttype == "function":
                fname = t.get("function", {}).get("name", "")
                if fname:
                    stripped_names.add(fname)
            elif ttype in ("web_search", "web_search_preview", "code_interpreter"):
                stripped_names.add(ttype)

    data["tools"] = kept
    removed = original - len(kept)

    # Sanitize tool_choice if it references a stripped tool
    tc = data.get("tool_choice")
    if tc and isinstance(tc, dict):
        tc_func = tc.get("function", {})
        tc_name = tc_func.get("name", "")
        if tc_name and tc_name in stripped_names:
            data["tool_choice"] = "auto"
        elif tc.get("type") in ("tool",) and tc.get("name") in stripped_names:
            data["tool_choice"] = "auto"

    if removed > 0:
        detail = ", ".join(
            f"{count}×{ttype}" for ttype, count in sorted(removed_types.items())
        )
        return data, f"stripped {removed} unsupported tool(s) ({original}→{len(kept)}): {detail}"
    return data, ""


async def handle_messages(request: web.Request) -> web.StreamResponse:
    """Handle /v1/messages (Anthropic API) by converting to OpenAI format."""
    body = await request.read()
    if not body:
        return web.Response(status=400, text="Empty body")

    try:
        anthropic_data = json.loads(body)
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    model = anthropic_data.get("model", "")
    is_stream = anthropic_data.get("stream", False)

    # Convert to OpenAI format
    openai_data = anthropic_to_openai(anthropic_data)

    # Strip unsupported tools
    openai_data, tool_log = filter_tools(openai_data)
    if tool_log:
        logger.info("POST /v1/messages: %s", tool_log)

    # Build forward headers: replace x-api-key with Authorization
    forward_headers = {
        "Content-Type": "application/json",
    }
    auth_token = request.headers.get("x-api-key") or request.headers.get("Authorization", "")
    if auth_token and not auth_token.startswith("Bearer "):
        forward_headers["Authorization"] = f"Bearer {auth_token}"
    elif auth_token:
        forward_headers["Authorization"] = auth_token

    target = f"{ROUTER_URL}/v1/chat/completions"
    encoded = json.dumps(openai_data).encode()

    logger.info("POST /v1/messages → /v1/chat/completions (stream=%s, model=%s)", is_stream, model)

    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(target, headers=forward_headers, data=encoded) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error("Upstream error %d: %s", resp.status, error_body[:500])
                    # Convert OpenAI error to Anthropic error format
                    return web.Response(
                        status=resp.status,
                        content_type="application/json",
                        text=json.dumps({
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": error_body[:1000],
                            },
                        }),
                    )

                if is_stream:
                    return await _stream_anthropic(request, resp, model)
                else:
                    openai_resp = await resp.json()
                    anthropic_resp = openai_to_anthropic(openai_resp, model)
                    return web.Response(
                        status=200,
                        content_type="application/json",
                        text=json.dumps(anthropic_resp),
                    )
    except aiohttp.ClientConnectorError as exc:
        logger.error("Cannot reach Rust router: %s", exc)
        return web.Response(status=502, text=json.dumps({
            "type": "error",
            "error": {"type": "api_error", "message": f"Bad Gateway: {exc}"},
        }))


async def _stream_anthropic(
    request: web.Request, resp: aiohttp.ClientResponse, model: str
) -> web.StreamResponse:
    """Convert OpenAI SSE stream to Anthropic SSE stream."""
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    converter = AnthropicStreamConverter(model)

    try:
        async for line in resp.content:
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    final = converter.finalize()
                    if final:
                        await response.write(final.encode())
                    break
                try:
                    chunk = json.loads(data_str)
                    converted = converter.convert_chunk(chunk)
                    if converted:
                        await response.write(converted.encode())
                except json.JSONDecodeError:
                    continue
        else:
            # Stream ended without [DONE]
            final = converter.finalize()
            if final:
                await response.write(final.encode())
    except Exception as exc:
        logger.error("Stream conversion error: %s", exc)

    await response.write_eof()
    return response


async def handle_responses(request: web.Request) -> web.StreamResponse:
    """Handle /v1/responses (OpenAI Responses API) by converting to Chat Completions."""
    body = await request.read()
    if not body:
        return web.Response(status=400, text="Empty body")

    try:
        resp_data = json.loads(body)
    except json.JSONDecodeError:
        return web.Response(status=400, text="Invalid JSON")

    model = resp_data.get("model", "")
    is_stream = resp_data.get("stream", False)

    # Convert Responses API → Chat Completions
    chat_data = responses_to_chat(resp_data)

    # Debug: log the converted messages summary
    msg_summary = "; ".join(
        f"{m.get('role','?')}:{str(m.get('content',''))[:80]}"
        for m in chat_data.get("messages", [])
    )
    logger.info("POST /v1/responses: input_type=%s, converted msgs: %s",
                type(resp_data.get("input")).__name__, msg_summary[:300])

    # Strip unsupported tools
    chat_data, tool_log = filter_tools(chat_data)
    if tool_log:
        logger.info("POST /v1/responses: %s", tool_log)

    # Forward headers
    forward_headers = {
        "Content-Type": "application/json",
        "Authorization": request.headers.get("Authorization", ""),
    }

    target = f"{ROUTER_URL}/v1/chat/completions"
    encoded = json.dumps(chat_data).encode()

    logger.info("POST /v1/responses → /v1/chat/completions (stream=%s, model=%s)", is_stream, model)

    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(target, headers=forward_headers, data=encoded) as resp:
                if resp.status != 200:
                    error_body = await resp.text()
                    logger.error("Upstream error %d: %s", resp.status, error_body[:500])
                    return web.Response(
                        status=resp.status,
                        content_type="application/json",
                        text=json.dumps({
                            "error": {
                                "message": error_body[:1000],
                                "type": "invalid_request_error",
                                "code": "json_parse_error",
                            },
                        }),
                    )

                if is_stream:
                    return await _stream_responses(request, resp, model)
                else:
                    chat_resp = await resp.json()
                    responses_resp = chat_to_responses(chat_resp, model)
                    return web.Response(
                        status=200,
                        content_type="application/json",
                        text=json.dumps(responses_resp),
                    )
    except aiohttp.ClientConnectorError as exc:
        logger.error("Cannot reach Rust router: %s", exc)
        return web.Response(status=502, text=json.dumps({
            "error": {"message": f"Bad Gateway: {exc}", "type": "api_error"},
        }))


async def _stream_responses(
    request: web.Request, resp: aiohttp.ClientResponse, model: str
) -> web.StreamResponse:
    """Convert OpenAI Chat Completions SSE stream to Responses API SSE stream."""
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await response.prepare(request)

    converter = ResponsesStreamConverter(model)

    try:
        async for line in resp.content:
            line = line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    final = converter.finalize()
                    if final:
                        await response.write(final.encode())
                    break
                try:
                    chunk = json.loads(data_str)
                    converted = converter.convert_chunk(chunk)
                    if converted:
                        await response.write(converted.encode())
                except json.JSONDecodeError:
                    continue
        else:
            final = converter.finalize()
            if final:
                await response.write(final.encode())
    except Exception as exc:
        logger.error("Responses stream conversion error: %s", exc)

    await response.write_eof()
    return response


async def handle_default(request: web.Request) -> web.StreamResponse:
    """Default handler: filter tools and forward to Rust router."""
    path = request.path
    method = request.method
    body = await request.read()

    if body and method in ("POST", "PUT", "PATCH"):
        try:
            data = json.loads(body)
            data, tool_log = filter_tools(data)
            if tool_log:
                logger.info("%s %s: %s", method, path, tool_log)
            body = json.dumps(data).encode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }

    target = f"{ROUTER_URL}{path}"
    if request.query_string:
        target += f"?{request.query_string}"

    try:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method=method,
                url=target,
                headers=headers,
                data=body if body else None,
                allow_redirects=False,
                auto_decompress=False,
            ) as resp:
                response = web.StreamResponse(
                    status=resp.status,
                    headers={
                        k: v for k, v in resp.headers.items()
                        if k.lower() not in (
                            "transfer-encoding",
                            "content-encoding",
                            "content-length",
                        )
                    },
                )
                await response.prepare(request)
                async for chunk in resp.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
    except aiohttp.ClientConnectorError as exc:
        logger.error("Cannot reach Rust router at %s: %s", ROUTER_URL, exc)
        return web.Response(status=502, text=f"Bad Gateway: {exc}")


async def handle(request: web.Request) -> web.StreamResponse:
    """Route requests to appropriate handler."""
    path = request.path

    # Anthropic Messages API → convert to OpenAI Chat Completion
    if path == "/v1/messages" and request.method == "POST":
        return await handle_messages(request)

    # OpenAI Responses API → convert to Chat Completions
    if path == "/v1/responses" and request.method == "POST":
        return await handle_responses(request)

    # All other paths: filter tools and forward
    return await handle_default(request)


async def wait_for_router(max_retries: int = 90, interval: float = 2.0) -> bool:
    """Poll Rust router /health until it responds 200."""
    url = f"{ROUTER_URL}/health"
    async with aiohttp.ClientSession() as session:
        for i in range(max_retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        logger.info("Rust router healthy (attempt %d)", i + 1)
                        return True
            except Exception:
                pass
            await asyncio.sleep(interval)
    return False


async def main() -> None:
    logger.info("Waiting for Rust router on port %s ...", ROUTER_PORT)
    if not await wait_for_router():
        logger.error("Rust router not ready after timeout — exiting")
        sys.exit(1)

    app = web.Application(client_max_size=MAX_BODY)
    app.router.add_route("*", "/{tail:.*}", handle)

    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PROXY_PORT)

    logger.info("Proxy listening on 0.0.0.0:%d → %s", PROXY_PORT, ROUTER_URL)
    logger.info("  /v1/messages → /v1/chat/completions (Anthropic→OpenAI conversion)")
    logger.info("  Other paths → direct forward with tool filtering")
    await site.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down proxy")
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
