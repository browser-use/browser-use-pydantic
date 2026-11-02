"""browser-use style API against a real headless Chrome, scripted model (no provider calls)."""

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
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from browser_use_pydantic import Agent, Browser, build_agent
from browser_use_pydantic.agent import model_ref

CHROME = os.environ.get('CHROME', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')


@pytest.fixture(scope='module')
def cdp_url():
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
            urllib.request.urlopen(f'http://127.0.0.1:{port}/json/version')
            break
        except OSError:
            time.sleep(0.1)
    os.environ['BH_TAB_MARKER'] = '0'
    yield f'http://127.0.0.1:{port}'
    proc.kill()
    shutil.rmtree(profile, ignore_errors=True)


def scripted(*turns):
    """Each turn is a list of (tool, args). Records every request's messages."""
    seen = []

    def fn(messages, info: AgentInfo):
        seen.append(messages)
        return ModelResponse(parts=[ToolCallPart(name, args) for name, args in turns[len(seen) - 1]])

    return FunctionModel(fn), seen


def prompts(messages):
    return [p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart)]


async def test_run_follow_up_and_close(cdp_url, tmp_path):
    model, seen = scripted(
        [('python', {'code': "new_tab('data:text/html,<title>Hi there</title>'); wait_for_load(); title = page_info()['title']"})],
        [('finish_from_repl', {'expression': 'title'})],
        [('finish', {'result': 'second answer'})],
    )
    browser = Browser(cdp_url=cdp_url)
    async with Agent('Read the title', model, browser=browser, workspace=tmp_path) as agent:
        first = await agent.run()
        assert first.final_result() == 'Hi there' and first.is_done() and first.structured_output is None
        agent.add_new_task('Now something else')
        second = await agent.run()
        assert second.final_result() == 'second answer'
        assert prompts(seen[-1]) == ['Read the title', 'Now something else']  # same transcript
        name = browser.name
    from browser_harness import admin
    assert browser.name is None and not admin.daemon_alive(name)


async def test_structured_output(cdp_url, tmp_path):
    class Story(BaseModel):
        title: str
        points: int

    model, _ = scripted([('finish', {'result': {'title': 'x', 'points': 3}})])
    async with Agent('t', model, browser=Browser(cdp_url=cdp_url), output_model_schema=Story,
                     workspace=tmp_path) as agent:
        r = await agent.run()
    assert r.structured_output == Story(title='x', points=3)


async def test_unfinished_run_has_no_final_result(cdp_url, tmp_path):
    model, _ = scripted(*[[('python', {'code': '1'})]] * 3)
    async with Agent('t', model, browser=Browser(cdp_url=cdp_url), workspace=tmp_path) as agent:
        r = await agent.run(max_steps=2)
    assert r.final_result() is None and not r.is_done()


def test_model_ids():
    assert model_ref('openai/gpt-5.6-luna') == 'openai-responses:gpt-5.6-luna'
    assert model_ref('anthropic/claude-opus-5-5') == 'anthropic:claude-opus-5-5'
    assert model_ref('openrouter/openai/gpt-5.6-luna') == 'openrouter:openai/gpt-5.6-luna'
    assert model_ref('openai-responses:gpt-6-luna') == 'openai-responses:gpt-6-luna'


def test_openai_settings_unchanged():
    """The evaluated configuration must not drift."""
    agent = build_agent('openai-responses:gpt-6-luna', reasoning_effort='medium')
    assert agent.model_settings == {'openai_reasoning_effort': 'medium', 'openai_reasoning_summary': 'auto',
                                    'openai_reasoning_context': 'auto', 'max_tokens': 32_768}
