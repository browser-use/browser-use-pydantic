"""Experimental code backend: `run_code` in Pydantic's Monty sandbox (pydantic-ai-harness CodeMode) instead of the
CPython REPL worker. Everything else (Session, BrowserLoop, delivery, prompt evidence rules) is shared with agent.py.

Shape:
- Host: one sandbox-callable tool, `harness_call(helper, args, kwargs)`, runs a browser-harness helper in a thread
  (helpers are sync IPC calls to the BU_NAME daemon), plus screenshot/show_image/checkpoint like worker.py.
- Sandbox: a prelude defines every helper with its browser-harness signature. CodeMode tools are keyword-only and
  cannot take **params, so `cdp('DOM.getBoxModel', backendNodeId=n)` or `click_at_xy(x, y)` would not run otherwise.
- MontyCell (capability around run_code): prelude, wall-clock cell timeout, errors as tool output (as in the worker,
  never retries that could end the run), text + native images result, output truncation, eval observer hook.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import pprint
import threading
from pathlib import Path
from typing import Any

from pydantic_ai import BinaryContent, ModelRetry, RunContext, Tool, ToolReturn
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ToolRetryError
from pydantic_ai_harness import CodeMode
from pydantic_monty import NOT_HANDLED, MountDir, OSAccess

from .worker import CDP_TIMEOUT_S, MAX_IMAGES_PER_CELL, _remote_upload, fit_image

CELL_TIMEOUT_S = 120  # Monty's own per-snippet compute and sleep allowance; wall clock is Session.cell_timeout_s
HELPERS = ('new_tab', 'goto_url', 'wait_for_load', 'page_info', 'list_tabs', 'current_tab', 'switch_tab', 'close_tab',
           'ensure_real_tab', 'js', 'cdp', 'click_at_xy', 'type_text', 'press_key', 'scroll', 'wait_for_element',
           'wait_for_network_idle', 'upload_file', 'http_get', 'drain_events', 'iframe_target', 'wait', 'fill_input',
           'dispatch_key', 'activate_tab')
LOCAL = ('screenshot', 'show_image', 'checkpoint')
# Defines the helpers once per sandbox session (again after a reset); see _prime. The imports mirror the worker
# namespace, where `from browser_harness.helpers import *` also brings json, math, os, time, base64 and Path.
PRELUDE = ('import json, math, os, time, base64\nfrom pathlib import Path\n'
           + ''.join(f'def {n}(*args, **kwargs):\n    return harness_call(helper={n!r}, args=list(args), kwargs=kwargs)\n'
                     for n in HELPERS + LOCAL)
           + '_bu_ready = True\n')
TIMED_OUT = ('Cell timed out after {:g}s. The sandbox restarted: Python variables are lost; files, checkpoints and '
             'browser state survive. Actions may already have happened; inspect before retrying.')

_local = threading.local()  # the MontyRepl whose helper call is running on this thread (routes screenshots)
_installed = threading.Lock()


def _install_cdp_wrapper() -> None:
    """Same wrapper as worker.py: 20s default IPC timeout, and screenshots taken by a helper call reach the model.

    In-process, so screenshots are only captured on threads running a helper for a MontyRepl (not the eval observer)."""
    from browser_harness import helpers

    with _installed:
        if getattr(helpers.cdp, '_bu_monty', False):
            return
        raw_cdp = helpers.cdp

        def cdp(method, session_id=None, **params):
            params.setdefault('_response_timeout', CDP_TIMEOUT_S)
            result = raw_cdp(method, session_id, **params)
            repl = getattr(_local, 'repl', None)
            if repl is not None and method == 'Page.captureScreenshot' and isinstance(result, dict) and result.get('data'):
                fmt = params.get('format', 'png')
                repl.add_image(base64.b64decode(result['data']), f'image/{"jpeg" if fmt == "jpeg" else fmt}')
            return result

        cdp.__doc__ = raw_cdp.__doc__
        cdp._bu_monty = True
        helpers.cdp = cdp


class _Clock(OSAccess):
    """Clock, randomness and env (empty) for the sandbox; files only through the workspace mount."""

    def dispatch(self, function_name, args, kwargs=None, *, is_async=False):
        if function_name == 'open' or function_name.startswith('Path.'):
            return NOT_HANDLED  # outside the mount: PermissionError instead of a silent in-memory filesystem
        return super().dispatch(function_name, args, kwargs, is_async=is_async)


class MontyRepl:
    """Monty counterpart of agent.Repl: workspace/state paths and per-cell host state. The sandbox REPL itself lives
    in CodeMode's per-run session; nothing to start or kill."""

    def __init__(self, workspace: Path, env: dict[str, str] | None = None, mode: str | None = 'default'):
        from .agent import check_mode

        if check_mode(mode) != 'default':
            raise ValueError("mode='ultrafast' is not implemented for the Monty backend")
        self.workspace, self.env, self.n, self.mode = workspace, env or {}, 0, 'default'
        self.state = workspace / '.browser-use'
        self.images: list[dict] = []
        self.notes: list[str] = []
        _install_cdp_wrapper()
        if 'BU_NAME' in self.env:  # helpers read the daemon name once at import; the worker got it via its env
            from browser_harness import helpers
            helpers.NAME = self.env['BU_NAME']

    async def close(self) -> None:
        pass

    def begin_cell(self) -> None:
        self.n += 1
        self.images, self.notes = [], []

    def add_image(self, data: bytes, media_type: str = 'image/png') -> bool:
        if len(self.images) >= MAX_IMAGES_PER_CELL:
            self.notes.append(f'[image dropped: {MAX_IMAGES_PER_CELL} per call max]')
            return False
        fitted = fit_image(data, media_type)
        if isinstance(fitted, str):
            self.notes.append(fitted)
            return False
        data, media_type = fitted
        self.images.append({'data': base64.b64encode(data).decode(), 'media_type': media_type})
        return True

    def path(self, p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else self.workspace / p

    def dispatch(self, helper: str, args: list, kwargs: dict) -> Any:
        from browser_harness import helpers

        _local.repl = self
        try:
            if helper in LOCAL:
                return getattr(self, helper)(*args, **kwargs)
            if helper == 'upload_file':  # the worker's cwd was the workspace; resolve relative paths the same way
                args = list(args)
                if len(args) > 1:
                    args[1] = str(self.path(args[1])) if isinstance(args[1], str) else [str(self.path(p)) for p in args[1]]
                elif 'path' in kwargs:
                    kwargs = {**kwargs, 'path': str(self.path(kwargs['path']))}
            if helper == 'upload_file' and self.env.get('BU_REMOTE_BROWSER') == '1':
                return _remote_upload(helpers.js)(*args, **kwargs)  # as the worker: local paths -> bytes
            return getattr(helpers, helper)(*args, **kwargs)
        finally:
            _local.repl = None

    # The three workspace helpers, as in worker.py.
    def screenshot(self, full=False):
        from browser_harness import helpers

        shots = self.state / 'screenshots'
        shots.mkdir(parents=True, exist_ok=True)
        r = helpers.cdp('Page.captureScreenshot', format='jpeg', quality=70, captureBeyondViewport=full)  # as worker
        path = shots / f'{len(list(shots.iterdir())):04d}.jpg'
        path.write_bytes(base64.b64decode(r['data']))
        return str(path)

    def show_image(self, path):
        p = self.path(path)
        ext = p.suffix.lower().lstrip('.').replace('jpg', 'jpeg')
        return self.add_image(p.read_bytes(), f'image/{ext}')

    def checkpoint(self, filename, value, partial=False):
        path = self.path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1, default=str))
        os.replace(tmp, path)
        if partial:
            self.state.mkdir(parents=True, exist_ok=True)
            (self.state / 'partial.json').write_text(json.dumps({'file': str(filename), 'value': value}, default=str))
        return str(filename)


async def harness_call(ctx: RunContext[Any], helper: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
    """Backend of the pre-defined browser helpers. Call the helpers (js, cdp, screenshot, ...) directly instead."""
    if helper not in HELPERS + LOCAL:
        raise ValueError(f'Unknown helper {helper!r}')
    return await asyncio.to_thread(ctx.deps.repl.dispatch, helper, args, kwargs)


def code_mode(workspace: Path) -> CodeMode:
    ws = str(workspace.resolve())
    return CodeMode(
        max_retries=1_000_000,  # errors are returned as output by MontyCell; this only bounds arg-validation retries
        max_tool_calls=100_000,  # the worker has no per-cell call cap
        mount=MountDir(virtual_path=ws, host_path=ws, mode='read-write'),  # same absolute paths inside and out
        os_access=_Clock(max_sleep=None),
        resource_limits={'max_duration_secs': CELL_TIMEOUT_S, 'max_memory': 2 << 30, 'max_suspensions': 2**40},
    )


async def _run_code(ctx: RunContext[Any], code: str, restart: bool = False) -> Any:
    """Run one snippet through CodeMode's own run_code tool (same session as the model's calls)."""
    tool = ctx.tool_manager.tools['run_code']
    return await tool.toolset.call_tool('run_code', {'code': code, 'restart': restart}, ctx, tool)


async def _prime(ctx: RunContext[Any], restart: bool = False) -> None:
    """Define the helpers if this sandbox session lacks them (first cell, after a timeout/limit reset or restart).

    The probe is a separate snippet because CodeMode type-checks the first snippet of a fresh session, which rejects
    any reference to a not-yet-defined name; the prelude itself then becomes that first, type-checked snippet, so the
    model's own snippets are never statically checked (the worker never did that either)."""
    if not restart:
        try:
            await _run_code(ctx, '_bu_ready')
            return
        except ModelRetry:
            pass
    await _run_code(ctx, PRELUDE, restart=restart)


def _retry_text(e: BaseException) -> str:
    cause = e.__cause__ if isinstance(e, ToolRetryError) else e
    return str(getattr(cause, 'message', cause))


def _render(value: Any) -> str:
    """CodeMode's return shapes -> worker-style text: printed output, then the last expression."""
    if isinstance(value, ToolReturn):
        value = value.return_value
    if isinstance(value, dict) and set(value) in ({'output'}, {'output', 'result'}):
        rest = _render(value['result']) if 'result' in value else ''
        return f'{value["output"]}{rest}'
    if value is None or value == {}:
        return ''
    if isinstance(value, str):
        return value
    return pprint.pformat(value, width=120, sort_dicts=False)  # as worker._display: any value, e.g. tuple dict keys


async def evaluate(ctx: RunContext[Any], expression: str, timeout: float) -> dict[str, Any]:
    """finish_from_repl for Monty: evaluate the expression in the same sandbox session. Worker 'eval' result shape."""
    try:
        async with asyncio.timeout(timeout):
            await _prime(ctx)
            r = await _run_code(ctx, expression)
    except TimeoutError:
        return {'value': None, 'error': f'Expression timed out after {timeout:g}s'}
    except Exception as e:  # noqa: BLE001 - ModelRetry from CodeMode carries the Monty traceback
        return {'value': None, 'error': _retry_text(e)}
    value = r.return_value if isinstance(r, ToolReturn) else r
    if isinstance(value, dict) and set(value) == {'output', 'result'}:  # the expression printed; deliver its value
        value = value['result']
    try:  # same encoding as the worker's eval; a value JSON can't encode is an error for the model, not a crash
        return {'value': value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str),
                'error': None}
    except (TypeError, ValueError) as e:
        return {'value': None, 'error': f'{type(e).__name__}: {e}'}


class MontyCell(AbstractCapability[Any]):
    """Make run_code behave like the worker's python tool (see module docstring)."""

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        if tool_def.name != 'run_code':
            return await handler(args)
        from .agent import MAX_OUTPUT_CHARS

        s = ctx.deps
        repl: MontyRepl = s.repl
        repl.begin_cell()
        text = error = None
        try:
            async with asyncio.timeout(s.cell_timeout_s) as deadline:
                await _prime(ctx, restart=bool(args.get('restart')))
                text = _render(await handler({**args, 'restart': False}))
        except (ToolRetryError, ModelRetry) as e:  # snippet error: CodeMode already prepended the printed output
            error = _retry_text(e)
        except TimeoutError:
            if not deadline.expired():
                raise
            # CodeMode reset the session on cancel. A helper already running in its thread finishes on its own
            # (bounded by its IPC timeout); printed output of the cell is not recoverable.
            error = TIMED_OUT.format(s.cell_timeout_s)
        except Exception as e:  # noqa: BLE001 - as in the worker, every failure is cell output, never the end of the run
            error = f'{type(e).__name__}: {e}'
        text = '\n'.join([text or '', *repl.notes]).strip('\n')
        cells = repl.state / 'cells'
        cells.mkdir(parents=True, exist_ok=True)
        out = cells / f'{repl.n:04d}.txt'
        out.write_text(f'{text}\nError: {error}' if error else text)
        r = {'output': text, 'error': error, 'images': list(repl.images), 'output_file': str(out)}
        if len(text) > MAX_OUTPUT_CHARS:
            text = f'{text[:MAX_OUTPUT_CHARS]}\n[Truncated. Full captured output: {out}]'
        if error:
            text = f'{text}\nError: {error}'.strip()
        if s.after_tool:
            await s.after_tool('run_code', {'code': args['code']}, r)
        shots = [BinaryContent(base64.b64decode(i['data']), media_type=i['media_type']) for i in r['images']]
        return [text or '(no output)', *shots]


def monty_toolset_tools() -> list[Tool]:
    return [Tool(harness_call, sequential=True)]  # sequential -> rendered as a plain def: called without await
