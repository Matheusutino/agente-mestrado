from __future__ import annotations

import dataclasses
import json
import os
import sys
from typing import Any

from dotenv import load_dotenv

from src.pipeline.data import dataset_profile, discover_datasets
from src.pipeline.evaluation import evaluate_classifier
from src.pipeline.modeling import train_classifier
from src.pipeline.preprocessing import preprocess_dataset
from src.pipeline.report import generate_report
from src.pipeline.representation import build_representation
from src.pipeline.web import search_arxiv
from src.types import PipelineResult

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
PLANNER_SYSTEM_PROMPT = """You are a text classification pipeline agent.
Think through which dataset is most appropriate for the task, which column should be treated as text, which column should be treated as the label, whether sparse lexical features or dense sentence embeddings best fit the context, which classifier is the strongest choice, and which metrics are worth reporting.
Do not ask questions. If information is ambiguous, make a reasonable assumption and record it.
Use the provided functions to inspect datasets, optionally consult arXiv when helpful, and execute the pipeline. If you choose sentence_transformer, provide a model_name explicitly.
When revision context is provided, inspect the prior attempt artifacts and either improve the next attempt or keep the current strategy if it is already adequate. Set stop_optimization=true only when no further attempt is warranted.
Return only a valid PipelineResult."""


PIPELINE_TOOLS = [
    discover_datasets,
    dataset_profile,
    search_arxiv,
    preprocess_dataset,
    build_representation,
    train_classifier,
    evaluate_classifier,
    generate_report,
]


@dataclasses.dataclass
class AgentExecutionRecord:
    result: PipelineResult
    all_messages_json: str
    new_messages_json: str
    usage: dict[str, Any]
    run_id: str | None
    conversation_id: str | None
    events: list[dict[str, Any]]


def build_planner_agent(model_name: str, thinking_effort: str | None = None):
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY is not set. Add it to a .env file or environment.")

    model_settings = None
    if thinking_effort is not None:
        try:
            from pydantic_ai.models.openai import OpenAIModelSettings

            model_settings = OpenAIModelSettings(reasoning_effort=thinking_effort)
        except (ImportError, TypeError):
            model_settings = {"thinking": thinking_effort}

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.profiles.openai import OpenAIModelProfile
        from pydantic_ai.providers.openai import OpenAIProvider

        provider = OpenAIProvider(
            base_url=NVIDIA_NIM_BASE_URL,
            api_key=api_key,
        )
        profile = OpenAIModelProfile(openai_supports_tool_choice_required=False)

        try:
            model = OpenAIModel(model_name=model_name, provider=provider, profile=profile)
        except TypeError:
            model = OpenAIModel(model_name, provider=provider)

        try:
            return Agent(
                model=model,
                output_type=PipelineResult,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                model_settings=model_settings,
                tools=PIPELINE_TOOLS,
            )
        except TypeError:
            return Agent(
                model=model,
                result_type=PipelineResult,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                model_settings=model_settings,
                tools=PIPELINE_TOOLS,
            )
    except Exception:
        pass

    try:
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIModel

        try:
            model = OpenAIModel(
                model_name=model_name,
                base_url=NVIDIA_NIM_BASE_URL,
                api_key=api_key,
            )
        except TypeError:
            model = OpenAIModel(model_name, base_url=NVIDIA_NIM_BASE_URL, api_key=api_key)

        try:
            return Agent(
                model=model,
                output_type=PipelineResult,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                model_settings=model_settings,
                tools=PIPELINE_TOOLS,
            )
        except TypeError:
            return Agent(
                model=model,
                result_type=PipelineResult,
                system_prompt=PLANNER_SYSTEM_PROMPT,
                model_settings=model_settings,
                tools=PIPELINE_TOOLS,
            )
    except Exception as exc:
        raise ImportError(
            "PydanticAI is required. Install a compatible version of pydantic-ai."
        ) from exc


def _extract_pipeline_output(result: Any) -> PipelineResult:
    for attribute in ("output", "data"):
        if hasattr(result, attribute):
            value = getattr(result, attribute)
            if isinstance(value, PipelineResult):
                return value
    if isinstance(result, PipelineResult):
        return result
    raise TypeError("Agent did not return a PipelineResult.")


def _serialize_usage(result: Any) -> dict[str, Any]:
    usage = result.usage() if hasattr(result, "usage") else None
    if usage is None:
        return {}
    if dataclasses.is_dataclass(usage):
        return dataclasses.asdict(usage)
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json")
    if hasattr(usage, "__dict__"):
        return dict(usage.__dict__)
    return {"value": str(usage)}


def _print_event(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message, file=sys.stderr, flush=True)


def _build_event_stream_handler(verbose: bool):
    from pydantic_ai import (
        FinalResultEvent,
        FunctionToolCallEvent,
        FunctionToolResultEvent,
        PartDeltaEvent,
        PartEndEvent,
        PartStartEvent,
    )
    from pydantic_ai.messages import (
        BuiltinToolCallEvent,
        BuiltinToolResultEvent,
        ThinkingPartDelta,
        ToolCallPart,
        ToolCallPartDelta,
    )

    text_started = False
    thinking_started = False
    event_log: list[dict[str, Any]] = []

    async def handle_event(_ctx, events) -> None:
        nonlocal text_started, thinking_started
        async for event in events:
            if isinstance(event, PartStartEvent):
                part_kind = getattr(event.part, "part_kind", "unknown")
                if isinstance(event.part, ToolCallPart):
                    payload = {
                        "event": "tool_part_start",
                        "tool_name": event.part.tool_name,
                        "args": str(event.part.args),
                    }
                    event_log.append(payload)
                    _print_event(
                        f"[planner/tool-part-start] tool_name={event.part.tool_name} args={event.part.args}",
                        verbose=verbose,
                    )
                elif part_kind == "text":
                    if not text_started:
                        event_log.append({"event": "text_start"})
                        _print_event("[planner/text] streaming model text:", verbose=verbose)
                        text_started = True
                elif part_kind == "thinking":
                    if not thinking_started:
                        event_log.append({"event": "thinking_start"})
                        _print_event("[planner/thinking]", verbose=verbose)
                        thinking_started = True
                else:
                    event_log.append({"event": "part_start", "kind": part_kind})
                    _print_event(f"[planner/part-start] kind={part_kind}", verbose=verbose)
                continue

            if isinstance(event, PartDeltaEvent):
                delta = event.delta
                if (
                    hasattr(delta, "content_delta")
                    and getattr(delta, "part_delta_kind", None) == "text"
                ):
                    content = delta.content_delta
                    event_log.append({"event": "text_delta", "content_delta": content})
                    if verbose:
                        print(content, end="", file=sys.stderr, flush=True)
                elif isinstance(delta, ThinkingPartDelta):
                    if delta.content_delta:
                        event_log.append(
                            {"event": "thinking_delta", "content_delta": delta.content_delta}
                        )
                        if verbose:
                            print(delta.content_delta, end="", file=sys.stderr, flush=True)
                elif isinstance(delta, ToolCallPartDelta):
                    event_log.append(
                        {"event": "tool_part_delta", "args_delta": str(delta.args_delta)}
                    )
                    _print_event(
                        f"[planner/tool-part-delta] args_delta={delta.args_delta}",
                        verbose=verbose,
                    )
                continue

            if isinstance(event, FunctionToolCallEvent):
                event_log.append(
                    {
                        "event": "tool_call",
                        "tool_name": event.part.tool_name,
                        "args": str(event.part.args),
                    }
                )
                _print_event(
                    f"[planner/tool-call] {event.part.tool_name}({event.part.args})",
                    verbose=verbose,
                )
                continue

            if isinstance(event, FunctionToolResultEvent):
                raw = getattr(event.result, "content", None) or event.content or ""
                preview = str(raw)
                event_log.append({"event": "tool_result", "content": preview})
                _print_event(
                    f"[planner/tool-result] {preview[:120].replace(chr(10), ' ')}",
                    verbose=verbose,
                )
                continue

            if isinstance(event, BuiltinToolCallEvent):
                part = getattr(event, "part", None)
                event_log.append({"event": "builtin_tool_call", "part": str(part)})
                _print_event(f"[planner/builtin-tool-call] part={part}", verbose=verbose)
                continue

            if isinstance(event, BuiltinToolResultEvent):
                event_log.append({"event": "builtin_tool_result", "content": str(event.content)})
                _print_event(
                    f"[planner/builtin-tool-result] content={event.content}",
                    verbose=verbose,
                )
                continue

            if isinstance(event, FinalResultEvent):
                event_log.append(
                    {
                        "event": "final_result",
                        "tool_name": event.tool_name,
                        "tool_call_id": event.tool_call_id,
                    }
                )
                if text_started and verbose:
                    print(file=sys.stderr, flush=True)
                _print_event(
                    f"[planner/final-result] tool_name={event.tool_name} tool_call_id={event.tool_call_id}",
                    verbose=verbose,
                )
                continue

            if isinstance(event, PartEndEvent):
                part_kind = getattr(event.part, "part_kind", "unknown")
                event_log.append({"event": "part_end", "kind": part_kind})
                if part_kind == "thinking" and verbose:
                    print(file=sys.stderr, flush=True)
                elif part_kind == "tool-call":
                    _print_event("[planner/tool-part-end]", verbose=verbose)

    return handle_event, event_log


def run_agent(agent: Any, prompt: str, verbose: bool) -> AgentExecutionRecord:
    handler, event_log = _build_event_stream_handler(verbose=verbose)

    if hasattr(agent, "run_sync"):
        result = agent.run_sync(prompt, event_stream_handler=handler)
    elif hasattr(agent, "run"):
        import asyncio

        result = asyncio.run(agent.run(prompt, event_stream_handler=handler))
    else:
        raise TypeError("Unsupported PydanticAI agent interface.")

    output = _extract_pipeline_output(result)
    all_messages_json = (
        result.all_messages_json().decode("utf-8") if hasattr(result, "all_messages_json") else "[]"
    )
    new_messages_json = (
        result.new_messages_json().decode("utf-8") if hasattr(result, "new_messages_json") else "[]"
    )

    return AgentExecutionRecord(
        result=output,
        all_messages_json=all_messages_json,
        new_messages_json=new_messages_json,
        usage=_serialize_usage(result),
        run_id=getattr(result, "run_id", None),
        conversation_id=getattr(result, "conversation_id", None),
        events=event_log,
    )
