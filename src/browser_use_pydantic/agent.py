"""Browser agent = Pydantic AI loop + one persistent Python REPL running browser-harness.

Learnings carried from browser-use-pi: deliver existing values (finish_from_repl), native screenshots
with only the last two kept, server-side compaction, one delivery turn near budget, partial results
that survive timeouts, and a killable worker so a hung cell never kills the run.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from genai_prices import Usage, calc_price
from pydantic import TypeAdapter, ValidationError
from pydantic_ai import Agent, BinaryContent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ThinkingPart, ToolReturnPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModelSettings
from pydantic_ai.usage import UsageLimits

from .prompt import SYSTEM_PROMPT

MAX_OUTPUT_CHARS = 12_000
KEEP_IMAGE_RESULTS = 2
DELIVER_NOW = (
    'Budget nearly exhausted. Deliver the verified result now with finish/finish_from_repl. '
    'Reference saved files; explicitly list missing evidence. Do not perform more browser actions.'
)


class Repl:
    """Parent side of worker.py. Kill-and-restart on timeout; files and the browser survive."""

    def __init__(self, workspace: Path, env: dict[str, str] | None = None):
        self.workspace, self.env, self.n = workspace, env or {}, 0
        self.proc: asyncio.subprocess.Process | None = None
        self.state = workspace / '.browser-use'

    async def start(self) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        parent, child = socket.socketpair()
        log = open(self.state / 'worker.log', 'ab')
        self.proc = await asyncio.create_subprocess_exec(
            sys.executable, '-m', 'browser_use_pydantic.worker', str(child.fileno()),
            cwd=self.workspace, pass_fds=[child.fileno()], stdin=asyncio.subprocess.DEVNULL,
            stdout=log, stderr=log, env={**os.environ, 'BH_TAB_MARKER': '0', **self.env, 'BU_STATE_DIR': str(self.state)},
        )
        child.close()
        log.close()
        self.reader, self.writer = await asyncio.open_connection(sock=parent, limit=256 * 1024 * 1024)
        ready = await asyncio.wait_for(self.reader.readline(), 60)
        if not ready:
            raise RuntimeError(f'REPL worker failed to start; see {self.state / "worker.log"}')

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.kill()
            await self.proc.wait()
        self.proc = None

    async def call(self, op: str, code: str, timeout: float) -> dict[str, Any]:
        if self.proc is None or self.proc.returncode is not None:
            await self.start()
        self.n += 1
        self.writer.write(json.dumps({'op': op, 'code': code, 'n': self.n}).encode() + b'\n')
        await self.writer.drain()
        try:
            line = await asyncio.wait_for(self.reader.readline(), timeout)
            if not line:
                raise ConnectionError('worker exited')
            return json.loads(line)
        except (TimeoutError, ConnectionError) as e:
            await self.close()
            out = self.state / 'cells' / f'{self.n:04d}.txt'
            why = f'Cell timed out after {timeout:g}s' if isinstance(e, TimeoutError) else 'REPL worker exited'
            return {'output': out.read_text(errors='replace') if out.exists() else '', 'images': [],
                    'output_file': str(out), 'value': None,
                    'error': f'{why}. The REPL restarted: Python variables are lost; files, checkpoints and '
                             'browser state survive. Actions may already have happened; inspect before retrying.'}


@dataclass
class Session:
    """Per-run state (Pydantic AI deps)."""

    repl: Repl
    timeout_s: float = 1700
    max_steps: int = 1000
    cell_timeout_s: float = 120
    started: float = field(default_factory=time.monotonic)
    finalizing: bool = False
    messages: list[ModelMessage] = field(default_factory=list)
    after_tool: Callable[[str, dict, dict], Awaitable[None]] | None = None  # observer hook (evals)

    def partial(self) -> Any:
        p = self.repl.state / 'partial.json'
        return json.loads(p.read_text())['value'] if p.exists() else None


async def python(ctx: RunContext[Session], code: str) -> list[Any]:
    """Run Python in the persistent REPL (browser-harness helpers and raw cdp() pre-imported).

    Top-level await works; variables survive calls; the last expression is printed. Cell deadline
    applies; checkpoint small batches before it. screenshot() returns native images.
    """
    r = await ctx.deps.repl.call('exec', code, ctx.deps.cell_timeout_s)
    text = r['output']
    if len(text) > MAX_OUTPUT_CHARS:
        text = f'{text[:MAX_OUTPUT_CHARS]}\n[Truncated. Full captured output: {r["output_file"]}]'
    if r['error']:
        text = f'{text}\nError: {r["error"]}'.strip()
    if ctx.deps.after_tool:
        await ctx.deps.after_tool('python', {'code': code}, r)
    shots = [BinaryContent(base64.b64decode(i['data']), media_type=i['media_type']) for i in r['images']]
    return [text or '(no output)', *shots]


def _outputs(output_type: Any) -> list[ToolOutput]:
    adapter = TypeAdapter(output_type)

    async def finish_from_repl(ctx: RunContext[Session], expression: str):
        r = await ctx.deps.repl.call('eval', expression, ctx.deps.cell_timeout_s)
        if r['error']:
            raise ModelRetry(f'Expression failed: {r["error"][-2000:]}')
        if output_type is str:
            return r['value']
        try:
            return adapter.validate_json(r['value'])
        except ValidationError as e:
            raise ModelRetry(f'Value does not match the result schema: {str(e)[:2000]}')

    def finish(result):
        return result

    finish.__annotations__ = {'result': output_type, 'return': output_type}
    finish_from_repl.__annotations__ = {'ctx': RunContext[Session], 'expression': str, 'return': output_type}
    return [
        ToolOutput(finish_from_repl, name='finish_from_repl', description=(
            'Deliver an existing value from the persistent REPL without printing or rewriting it. The Python '
            'expression must match the result schema of finish; for a string result use json.dumps(records) or an '
            'existing text variable. Evaluated once; ends the task after validation.')),
        ToolOutput(finish, name='finish', description=(
            'Submit the verified final result. For data already in the REPL, prefer finish_from_repl to avoid '
            'rewriting it. This ends the task.')),
    ]


class BrowserLoop(AbstractCapability[Session]):
    """Keep only recent screenshots; force one delivery turn near the time/step budget."""

    async def before_model_request(self, ctx: RunContext[Session], request_context: ModelRequestContext):
        s = ctx.deps
        s.messages = ctx.messages
        pruned = _prune_images(ctx.messages)
        if pruned is not None:  # same pattern as ProcessHistory: new objects, persisted
            ctx.messages[:] = pruned
            request_context.messages = pruned.copy()
        if not s.finalizing and (
            ctx.run_step >= s.max_steps - 1 or time.monotonic() - s.started >= 0.9 * s.timeout_s
        ):
            s.finalizing = True
            ctx.enqueue(DELIVER_NOW)
        return request_context

    def get_model_settings(self):
        # allowed_tools keeps the tool list (cached prefix) while forcing delivery.
        return lambda ctx: {'tool_choice': ['finish_from_repl', 'finish']} if ctx.deps.finalizing else {}


def _is_image(c: Any) -> bool:
    return isinstance(c, BinaryContent) and c.is_image


def _prune_images(messages: list[ModelMessage]) -> list[ModelMessage] | None:
    """Keep images only in the last KEEP_IMAGE_RESULTS tool results. Returns None when nothing changes."""
    out, seen, changed = list(messages), 0, False
    for i in range(len(out) - 1, -1, -1):
        m = out[i]
        if not isinstance(m, ModelRequest):
            continue
        parts = list(m.parts)
        for j, part in enumerate(parts):
            if isinstance(part, ToolReturnPart) and isinstance(part.content, list) and any(map(_is_image, part.content)):
                seen += 1
                if seen > KEEP_IMAGE_RESULTS:
                    content = [('[older screenshot removed]' if _is_image(c) else c) for c in part.content]
                    parts[j] = replace(part, content=content)
                    out[i] = m = replace(m, parts=parts)
                    changed = True
    return out if changed else None


def build_agent(
    model: str = 'openai-responses:gpt-6-luna',
    *,
    output_type: Any = str,
    reasoning_effort: str = 'medium',
    instructions: str = '',
    compact_at_tokens: int = 150_000,
) -> Agent[Session, Any]:
    settings: OpenAIResponsesModelSettings = {
        'openai_reasoning_effort': reasoning_effort,
        'openai_reasoning_summary': 'auto',
        'max_tokens': 32_768,
    }
    return Agent(
        model,
        deps_type=Session,
        defer_model_check=True,
        output_type=_outputs(output_type),
        instructions=f'{SYSTEM_PROMPT}\n{instructions}'.strip(),
        tools=[python],
        model_settings=settings,
        retries={'tools': 1, 'output': 3},
        capabilities=[BrowserLoop(), OpenAICompaction(token_threshold=compact_at_tokens)],
    )


@dataclass
class RunResult:
    status: str  # completed | timeout | max_steps | error
    text: str
    output: Any
    steps: int
    duration_s: float
    usage: dict[str, int]
    cost_usd: float
    partial: Any
    error: str | None
    messages: list[ModelMessage]


def usage_and_cost(messages: list[ModelMessage], model_ref: str) -> tuple[dict[str, int], float]:
    total = {'input': 0, 'output': 0, 'cache_read': 0, 'requests': 0}
    cost = 0.0
    for m in messages:
        if isinstance(m, ModelResponse):
            u = m.usage
            total['input'] += u.input_tokens
            total['output'] += u.output_tokens
            total['cache_read'] += u.cache_read_tokens
            total['requests'] += 1
            try:
                cost += float(calc_price(Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                                               cache_read_tokens=u.cache_read_tokens),
                                         model_ref=model_ref, provider_id='openai').total_price)
            except Exception:  # noqa: BLE001 - unknown price: report tokens, cost stays partial
                pass
    return total, cost


def last_text(messages: list[ModelMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, ModelResponse):
            text = '\n'.join(p.content for p in m.parts if isinstance(p, TextPart)).strip()
            if text:
                return text
    return ''


async def run(agent: Agent[Session, Any], task: str, session: Session, model_ref: str) -> RunResult:
    """Run one task with graceful budgets. Never raises for agent/provider failures."""
    status, output, error = 'completed', None, None
    try:
        async with asyncio.timeout(session.timeout_s):
            result = await agent.run(task, deps=session,
                                     usage_limits=UsageLimits(request_limit=session.max_steps))
        output, messages = result.output, result.all_messages()
    except TimeoutError:
        status, messages = 'timeout', session.messages
    except Exception as e:  # noqa: BLE001
        name = type(e).__name__
        status = 'max_steps' if name == 'UsageLimitExceeded' else 'error'
        error, messages = f'{name}: {e}', session.messages
    partial = session.partial()
    if output is None:
        output = json.dumps(partial, ensure_ascii=False) if partial is not None else last_text(messages)
    text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, default=str)
    usage, cost = usage_and_cost(messages, model_ref)
    steps = sum(isinstance(m, ModelResponse) for m in messages)
    return RunResult(status, text, output, steps, time.monotonic() - session.started, usage, cost,
                     partial, error, messages)


def thinking_and_text(m: ModelResponse) -> list[tuple[str, str]]:
    return [('thinking' if isinstance(p, ThinkingPart) else 'text', p.content)
            for p in m.parts if isinstance(p, (ThinkingPart, TextPart)) and p.content]
