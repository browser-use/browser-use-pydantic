"""End-to-end against a real isolated headless Chrome, scripted model (no provider calls)."""

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from browser_use_pydantic import Repl, Session, build_agent, run

CHROME = os.environ.get('CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')


@pytest.fixture(scope='module')
def browser():
    if not Path(CHROME).exists():
        pytest.skip('Chrome not installed')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    profile = tempfile.mkdtemp()
    proc = subprocess.Popen([CHROME, '--headless=new', f'--remote-debugging-port={port}', f'--user-data-dir={profile}',
                             '--no-first-run', 'about:blank'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        try:
            ws = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version'))['webSocketDebuggerUrl']
            break
        except OSError:
            time.sleep(0.1)
    name = f'bupt{port}'
    from browser_harness import admin
    os.environ['BH_TAB_MARKER'] = '0'  # daemon marks titles otherwise
    admin.ensure_daemon(name=name, env={'BU_CDP_WS': ws})
    yield {'BU_NAME': name}
    try:
        from browser_harness import _ipc as ipc
        c, token = ipc.connect(name, timeout=2)
        ipc.request(c, token, {'meta': 'shutdown'})
    except Exception:
        pass
    proc.kill()
    shutil.rmtree(profile, ignore_errors=True)


def script(*steps):
    """FunctionModel that plays tool calls in order and records what it saw."""
    seen = []

    def fn(messages, info: AgentInfo):
        seen.append(messages[-1])
        if len(seen) > len(steps):
            raise AssertionError(f'unexpected model call {len(seen)}; last {seen[-1]!r}'[:4000])
        name, args = steps[len(seen) - 1]
        return ModelResponse(parts=[ToolCallPart(name, args)])

    return FunctionModel(fn), seen


async def go(env, tmp_path, steps, **kw):
    model, seen = script(*steps)
    agent = build_agent()
    session = Session(repl=Repl(tmp_path, env), **kw)
    try:
        with agent.override(model=model):
            result = await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    return result, seen


async def test_browser_screenshot_and_finish_from_repl(browser, tmp_path):
    result, seen = await go(browser, tmp_path, [
        ('python', {'code': "new_tab('data:text/html,<title>T</title><h1>Hello</h1>')\nwait_for_load()\n"
                            "info = page_info()\nh = js('document.querySelector(\"h1\").textContent')\nscreenshot()\nh"}),
        ('python', {'code': 'records = [{"h": h, "title": info["title"]}]\nawait __import__("asyncio").sleep(0)\nlen(records)'}),
        ('finish_from_repl', {'expression': 'json.dumps(records)'}),
    ])
    assert result.status == 'completed', result.error
    assert json.loads(result.output) == [{'h': 'Hello', 'title': 'T'}]
    first = [p for p in seen[1].parts if isinstance(p, ToolReturnPart)][0]
    assert first.content[0].strip() == 'Hello'
    assert any(getattr(c, 'is_image', False) for c in first.content)  # native image to the model
    assert [p for p in seen[2].parts if isinstance(p, ToolReturnPart)][0].content[0].strip() == '1'


async def test_cell_timeout_restarts_worker_keeps_files(browser, tmp_path):
    result, seen = await go(browser, tmp_path, [
        ('python', {'code': "x = 1\ncheckpoint('rows.json', [1, 2], partial=True)\nprint('saved', flush=True)\nimport time\ntime.sleep(30)"}),
        ('python', {'code': "print('x' in globals(), json.load(open('rows.json')))"}),
        ('python', {'code': 'raise ValueError("boom")'}),
        ('finish', {'result': 'done'}),
    ], cell_timeout_s=3)
    assert result.output == 'done'
    timed_out = [p for p in seen[1].parts if isinstance(p, ToolReturnPart)][0].content[0]
    assert 'saved' in timed_out and 'timed out' in timed_out
    assert [p for p in seen[2].parts if isinstance(p, ToolReturnPart)][0].content[0].strip() == 'False [1, 2]'
    assert 'ValueError: boom' in [p for p in seen[3].parts if isinstance(p, ToolReturnPart)][0].content[0]


async def test_timeout_returns_partial(browser, tmp_path):
    result, _ = await go(browser, tmp_path, [
        ('python', {'code': "checkpoint('out.json', {'rows': 3}, partial=True)"}),
        ('python', {'code': 'import time\ntime.sleep(30)'}),
    ], timeout_s=4, cell_timeout_s=60)
    assert result.status == 'timeout'
    assert json.loads(result.text) == {'rows': 3}


async def test_only_last_two_screenshot_results_keep_images(browser, tmp_path):
    shot = ('python', {'code': "new_tab('data:text/html,<h1>x</h1>') if not list_tabs(False) else None\nscreenshot()\n'shot'"})
    result, seen = await go(browser, tmp_path, [shot, shot, shot, ('finish', {'result': 'ok'})])
    assert result.output == 'ok'
    returns = [p for m in result.messages for p in getattr(m, 'parts', []) if isinstance(p, ToolReturnPart)
               and p.tool_name == 'python']
    images = [sum(getattr(c, 'is_image', False) for c in p.content) for p in returns]
    assert images == [0, 1, 1]
    assert '[older screenshot removed]' in returns[0].content


async def test_parallel_python_calls_are_serialized(browser, tmp_path):
    # hard-high 0vgnxb/74rs7g/up8ijl: two python calls in one response crashed the run (readuntil race)
    calls = [0]

    def fn(messages, info: AgentInfo):
        calls[0] += 1
        if calls[0] == 1:
            return ModelResponse(parts=[ToolCallPart('python', {'code': 'import time; time.sleep(0.3); a = 1'}),
                                        ToolCallPart('python', {'code': 'b = 2'})])
        return ModelResponse(parts=[ToolCallPart('finish_from_repl', {'expression': 'str(a + b)'})])

    agent, session = build_agent(), Session(repl=Repl(tmp_path, browser))
    try:
        with agent.override(model=FunctionModel(fn)):
            result = await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    assert (result.status, result.output, result.error) == ('completed', '3', None)


async def test_oversized_and_invalid_images_never_reach_provider(browser, tmp_path):
    # v2 bub2-002 (43200 patches > 30000) and hard y0bvr6 (invalid image data) 400'd the whole run
    code = ("from PIL import Image; Image.new('RGB', (1440, 30000), 'white').save('tall.png'); "
            "open('bad.png', 'wb').write(b'\\x89PNG not really'); print(show_image('tall.png'), show_image('bad.png'))")
    result, seen = await go(browser, tmp_path, [('python', {'code': code}), ('finish', {'result': 'ok'})])
    part = next(p for p in seen[1].parts if isinstance(p, ToolReturnPart))
    images = [c for c in part.content if not isinstance(c, str)]
    assert result.status == 'completed' and len(images) == 1
    from PIL import Image
    import io
    assert max(Image.open(io.BytesIO(images[0].data)).size) <= 2048
    assert 'not a decodable image' in part.content[0]


async def test_workspace_path_is_in_instructions(browser, tmp_path):
    seen = []

    def fn(messages, info: AgentInfo):
        seen.append(info.instructions)
        return ModelResponse(parts=[ToolCallPart('finish', {'result': 'ok'})])

    agent, session = build_agent(), Session(repl=Repl(tmp_path, browser))
    try:
        with agent.override(model=FunctionModel(fn)):
            await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    assert json.dumps(str(tmp_path.resolve())) in seen[0] and 'never `await` them' in seen[0]


async def test_text_reply_gets_one_delivery_repair_turn(browser, tmp_path):
    # A text-only reply used to be impossible (tool_choice=required every turn). Now it is allowed, and ending the
    # work in text triggers bu-pi's single delivery-repair turn with delivery tools forced.
    from pydantic_ai.messages import RetryPromptPart, TextPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models._tool_choice import resolve_tool_choice
    from browser_use_pydantic.agent import DELIVERY_REPAIR

    seen = []

    def fn(messages, info: AgentInfo):
        params = ModelRequestParameters(function_tools=info.function_tools, output_tools=info.output_tools,
                                        allow_text_output=info.allow_text_output)
        seen.append((messages[-1], resolve_tool_choice(info.model_settings, params)))
        if len(seen) == 1:
            return ModelResponse(parts=[ToolCallPart('python', {'code': 'rows = [1, 2, 3]'})])
        if len(seen) == 2:
            return ModelResponse(parts=[TextPart('I found 3 rows.')])
        return ModelResponse(parts=[ToolCallPart('finish_from_repl', {'expression': 'json.dumps(rows)'})])

    agent, session = build_agent(), Session(repl=Repl(tmp_path, browser))
    try:
        with agent.override(model=FunctionModel(fn)):
            result = await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    assert [c for _, c in seen[:2]] == ['auto', 'auto']  # the model may stop and talk
    repair_msg, repair_choice = seen[2]
    assert any(DELIVERY_REPAIR in (p.model_response() if isinstance(p, RetryPromptPart) else str(getattr(p, 'content', '')))
               for p in repair_msg.parts)
    assert repair_choice == ('required', {'finish', 'finish_from_repl'})
    assert (result.status, result.output) == ('completed', '[1, 2, 3]')


def _choice_recorder(responses):
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models._tool_choice import resolve_tool_choice
    seen = []

    def fn(messages, info: AgentInfo):
        params = ModelRequestParameters(function_tools=info.function_tools, output_tools=info.output_tools,
                                        allow_text_output=info.allow_text_output)
        seen.append((messages, resolve_tool_choice(info.model_settings, params)))
        return responses[len(seen) - 1]

    return FunctionModel(fn), seen


async def test_budget_turn_forces_delivery_on_the_same_request(browser, tmp_path):
    # v1b let one more python call through after "Budget nearly exhausted" (settings resolved before the hook);
    # in 4n89qk/t4p501 that turn was spent and the forced finish came one turn late.
    from browser_use_pydantic.agent import DELIVER_NOW
    model, seen = _choice_recorder([ModelResponse(parts=[ToolCallPart('python', {'code': 'x = 1'})]),
                                    ModelResponse(parts=[ToolCallPart('finish', {'result': 'one'})])])
    agent, session = build_agent(), Session(repl=Repl(tmp_path, browser), max_steps=3)
    try:
        with agent.override(model=model):
            result = await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    assert seen[0][1] == 'auto'
    assert DELIVER_NOW in str(seen[1][0][-1]) and seen[1][1] == ('required', {'finish', 'finish_from_repl'})
    assert result.output == 'one'


@pytest.mark.parametrize('first', ['...', 'Saved: [leads](leads.csv)', 'Saved in the workspace: `leads.csv`'])
async def test_placeholder_or_missing_file_delivery_gets_one_repair(browser, tmp_path, first):
    # 4n89qk delivered finish("..."); t4p501 linked leads_verified.csv that was never written.
    model, seen = _choice_recorder([ModelResponse(parts=[ToolCallPart('finish', {'result': first})]),
                                    ModelResponse(parts=[ToolCallPart('finish', {'result': first})])])
    agent, session = build_agent(), Session(repl=Repl(tmp_path, browser))
    try:
        with agent.override(model=model):
            result = await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    retry = [p.content for p in seen[1][0][-1].parts if getattr(p, 'outcome', None) == 'retried']
    assert retry and ('placeholder' in retry[0] or 'leads.csv' in retry[0])
    assert (len(seen), result.status, result.output) == (2, 'completed', first)  # second identical finish accepted


async def test_existing_file_link_and_short_answer_pass(browser, tmp_path):
    (tmp_path / 'out.csv').write_text('a\n1\n')
    for answer in ['7', 'See [out.csv](out.csv), `out.csv` and [site](https://example.com)']:
        result, seen = await go(browser, tmp_path, [('finish', {'result': answer})])
        assert (len(seen), result.output) == (1, answer)


async def test_partial_result_with_budget_left_gets_one_continue_turn(browser, tmp_path):
    # v1b quit by choice with most of the budget left (2vxyzx, bub2-007, bub2-014).
    first = 'Partial: only 5 of 37 listings checked.'
    model, seen = _choice_recorder([ModelResponse(parts=[ToolCallPart('finish', {'result': first})]),
                                    ModelResponse(parts=[ToolCallPart('finish', {'result': first})])])
    agent, session = build_agent(), Session(repl=Repl(tmp_path, browser))
    try:
        with agent.override(model=model):
            result = await run(agent, 'task', session, 'gpt-6-luna')
    finally:
        await session.repl.close()
    retry = [p.content for p in seen[1][0][-1].parts if getattr(p, 'outcome', None) == 'retried']
    assert retry and 'minutes remain' in retry[0]
    assert (len(seen), result.output) == (2, first)


async def test_screenshot_is_jpeg(browser, tmp_path):
    # bu-pi parity: JPEG q70; v1b's 60s full-PNG captures timed out on 48 of 280 Hard screenshot() calls.
    result, seen = await go(browser, tmp_path, [
        ('python', {'code': "new_tab('data:text/html,<h1>hi</h1>')\nwait_for_load()\np = screenshot()\np"}),
        ('finish_from_repl', {'expression': 'p'})])
    out = [p for p in seen[1].parts if isinstance(p, ToolReturnPart)][0].content
    assert result.status == 'completed', out
    assert result.output.endswith('.jpg') and Path(result.output).read_bytes()[:2] == b'\xff\xd8'
    assert any(getattr(c, 'media_type', '') == 'image/jpeg' for c in out)
