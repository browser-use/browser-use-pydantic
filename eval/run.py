#!/usr/bin/env python3
"""Eval adapter: same executable contract, options, browser and evidence as browser-use-pi's eval/run.mjs.

One isolated Browser Use Cloud browser per task, a browser-harness daemon attached to it, the
Pydantic AI agent driving it. Writes the same files bu-pi writes so Laith and findings judge it identically.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

ALLOWED = {'reasoning_effort', 'max_context_chars', 'task_timeout_seconds', 'proxy_country_code',
           'browser_timeout_minutes', 'browser_allow_resizing', 'evidence_format', 'research_tools',
           'delivery_review'}
INSTRUCTIONS = {
    True: 'Use browser UI, public search and source APIs for research; use files/scripts for processing.',
    False: 'Use browser UI and page evaluation for research. Do not use web search.',
}
INSTRUCTIONS_TAIL = (' Do not read files outside the output workspace or inspect benchmark source, rubrics, judge code,'
                     ' or credentials. Save requested files incrementally in workspace.')


def parse_options(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError('options must be an object')
    unknown = set(value) - ALLOWED
    if unknown:
        raise ValueError(f'Unknown option: {sorted(unknown)[0]}')
    o = {'reasoning_effort': 'medium', 'max_context_chars': 800_000, 'task_timeout_seconds': 1700,
         'proxy_country_code': 'us', 'browser_timeout_minutes': 60, **value}
    if o['reasoning_effort'] not in {'minimal', 'low', 'medium', 'high', 'xhigh'}:
        raise ValueError('Invalid reasoning_effort')
    if not 1 <= o['task_timeout_seconds'] <= 7200:
        raise ValueError('task_timeout_seconds must be 1..7200')
    if o['browser_timeout_minutes'] * 60 <= o['task_timeout_seconds'] + 30:
        raise ValueError('Browser lifetime must exceed task timeout plus 30 seconds')
    if o.get('evidence_format') not in (None, 'findings'):
        raise ValueError('evidence_format must be findings when provided')
    if o.get('delivery_review'):
        raise ValueError('delivery_review is not implemented in this adapter')
    return o


def clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f'...[at least {len(text) - limit} characters omitted from the middle]...'
    budget = limit - len(marker)
    head = budget // 3
    return text[:head] + marker + text[-(budget - head):]


def cloud(path: str, method: str, body: dict) -> dict:
    req = urllib.request.Request(
        f'https://api.browser-use.com/api/v3{path}', method=method, data=json.dumps(body).encode(),
        headers={'X-Browser-Use-API-Key': os.environ['BROWSER_USE_API_KEY'], 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    return json.loads(data) if data else {}


async def main() -> int:
    env = os.environ
    workspace = Path(env['EVAL_WORKSPACE']).resolve()
    result_path = Path(env['EVAL_RESULT_PATH']).resolve()
    envelope = {'status': 'failed', 'final_output': '', 'self_reported_success': None,
                'error': 'Harness did not start', 'metrics': {}, 'artifacts': [], 'metadata': {}}
    browser = None
    session = None
    cleanup_errors: list[str] = []
    span_cm = None
    Laminar = None
    try:
        options = parse_options(json.loads(env.get('EVAL_OPTIONS_JSON') or '{}'))
        if int(env['EVAL_TIMEOUT_MINUTES']) * 60 < options['task_timeout_seconds'] + 90:
            raise ValueError('Platform timeout must exceed task timeout by at least 90 seconds')
        if not env.get('EVAL_MODEL_API_KEY') or not env.get('BROWSER_USE_API_KEY'):
            raise ValueError('Model and browser credentials are required')
        task = json.loads(Path(env['EVAL_TASK_PATH']).read_text())
        findings = options.get('evidence_format') == 'findings'
        model_name = env['EVAL_MODEL'].removeprefix('openai/')
        os.environ['OPENAI_API_KEY'] = env['EVAL_MODEL_API_KEY']
        bu_name = f'eval{os.getpid()}'
        os.environ.update(BU_NAME=bu_name, BH_TAB_MARKER='0', BH_OPEN_LIVE_URL='0', PYDANTIC_AI_NO_BANNER='1')

        try:
            from lmnr import Laminar as _Laminar

            if env.get('LMNR_PROJECT_API_KEY'):
                _Laminar.initialize(project_api_key=env['LMNR_PROJECT_API_KEY'])
                Laminar = _Laminar
                span_cm = Laminar.start_as_current_span(
                    name='pydantic.browser_harness', input={'task': task, 'options': options, 'model': model_name},
                    parent_span_context=env.get('LMNR_SPAN_CONTEXT') or None)
                span_cm.__enter__()
        except Exception as e:  # telemetry must never fail the task
            print(f'Laminar unavailable: {e}', file=sys.stderr)

        from browser_harness import admin
        from browser_harness import helpers as bh

        from browser_use_pydantic import Repl, Session, build_agent, run
        from browser_use_pydantic.agent import thinking_and_text
        from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelResponse

        output_dir = workspace / 'agent_outputs'
        output_dir.mkdir(parents=True, exist_ok=True)
        shots_dir = workspace / 'judge_screenshots'
        shots_dir.mkdir(parents=True, exist_ok=True)

        body = {'timeout': options['browser_timeout_minutes'], 'proxyCountryCode': options['proxy_country_code'],
                'browserScreenWidth': 1440, 'browserScreenHeight': 900, 'enableRecording': False}
        if 'browser_allow_resizing' in options:
            body['allowResizing'] = options['browser_allow_resizing']
        browser = cloud('/browsers', 'POST', body)
        if not browser.get('id') or not browser.get('cdpUrl'):
            raise RuntimeError('Browser provider returned no browser id/CDP endpoint')
        ws = json.load(urllib.request.urlopen(f'{browser["cdpUrl"]}/json/version', timeout=30))['webSocketDebuggerUrl']
        await asyncio.to_thread(admin.ensure_daemon, name=bu_name, env={'BU_CDP_WS': ws})

        steps: list[str] = []
        shots: list[str] = []
        shot_steps: list[int] = []
        shot_errors: list[str] = []
        shot_time = [0.0]
        seen_responses = [0]
        events = open(workspace / 'events.jsonl', 'a')
        agent_steps = open(workspace / 'agent_steps.txt', 'a')

        def capture() -> bytes:
            params = {'format': 'png'} if findings else {'format': 'jpeg', 'quality': 70}
            import base64
            return base64.b64decode(bh.cdp('Page.captureScreenshot', _response_timeout=15, **params)['data'])

        async def after_tool(name: str, args: dict, r: dict) -> None:
            for m in session.messages[seen_responses[0]:]:
                if isinstance(m, ModelResponse):
                    for kind, content in thinking_and_text(m):
                        steps.append(f'{kind}: {clip(content, 4000)}')
                        events.write(json.dumps({'type': 'message', 'kind': kind, 'content': content}) + '\n')
            seen_responses[0] = len(session.messages)
            result = {'output': r['output'], 'error': r['error'], 'images': len(r['images'])}
            steps.append(f'{name}: {clip(json.dumps(args), 2000)}\nresult: {clip(json.dumps(result), 20000)}')
            events.write(json.dumps({'type': 'tool', 'name': name, 'args': args, 'result': result}) + '\n')
            agent_steps.write(f'{name}{" ERROR" if r["error"] else ""}\n{json.dumps(result)}\n')
            events.flush()
            agent_steps.flush()
            started = time.monotonic()
            try:  # observer screenshot of the agent's attached tab, for the judge only
                data = await asyncio.wait_for(asyncio.to_thread(capture), 15)
                shot = shots_dir / f'{len(list(shots_dir.iterdir())) + 1:03d}.{"png" if findings else "jpg"}'
                shot.write_bytes(data)
                if findings:
                    shots.append(str(shot.relative_to(workspace)))
                    shot_steps.append(len(steps))
            except Exception as e:  # noqa: BLE001
                if len(shot_errors) < 20:
                    shot_errors.append(str(e)[:500])
            shot_time[0] += time.monotonic() - started

        agent = build_agent(
            f'openai-responses:{model_name}', reasoning_effort=options['reasoning_effort'],
            instructions=INSTRUCTIONS[findings] + INSTRUCTIONS_TAIL,
            compact_at_tokens=int(options['max_context_chars'] * 0.75 / 4))
        session = Session(repl=Repl(output_dir, {'BU_NAME': bu_name}), timeout_s=options['task_timeout_seconds'],
                          max_steps=int(env.get('EVAL_MAX_STEPS') or 35), cell_timeout_s=120,
                          after_tool=after_tool)
        prompt = task['confirmed_task'] + (f'\nStarting website: {task["website"]}' if task.get('website') else '')
        res = await run(agent, prompt, session, model_name)
        events.close()
        agent_steps.close()

        evidence = {}
        if findings:
            ev_path = workspace / 'findings-evidence.json'
            subprocess.run([sys.executable, str(Path(__file__).with_name('findings.py')), str(output_dir), str(ev_path)],
                           check=True, timeout=60)
            evidence = {**json.loads(ev_path.read_text()), 'steps': steps, 'judge_screenshots': shots,
                        'judge_screenshot_steps': shot_steps, 'timed_out': res.status == 'timeout'}
        (workspace / 'final_message.txt').write_text(res.text)
        (workspace / 'messages.json').write_bytes(ModelMessagesTypeAdapter.dump_json(res.messages, indent=1))
        (workspace / 'sdk-result.json').write_text(json.dumps({
            'status': res.status, 'text': res.text, 'steps': res.steps, 'usage': res.usage, 'cost': res.cost_usd,
            'partial': res.partial, 'error': res.error, 'duration_s': res.duration_s}, indent=2, default=str))
        files = sorted(str(p.relative_to(workspace)) for d in (output_dir, shots_dir) for p in d.rglob('*')
                       if p.is_file() and '.browser-use' not in p.parts)
        envelope = {
            'status': 'failed' if res.status == 'error' else 'completed',
            'final_output': res.text,
            'self_reported_success': None,
            'error': res.error,
            'metrics': {'steps': res.steps, 'duration_seconds': res.duration_s, 'input_tokens': res.usage['input'],
                        'output_tokens': res.usage['output'], 'cached_input_tokens': res.usage['cache_read'],
                        'cache_write_tokens': 0, 'total_tokens': res.usage['input'] + res.usage['output'],
                        'total_cost': res.cost_usd},
            'artifacts': files + ['events.jsonl', 'agent_steps.txt', 'final_message.txt', 'sdk-result.json',
                                  'messages.json'],
            'metadata': {**evidence, 'stop_reason': res.status, 'browser': {'id': browser['id']}, 'options': options,
                         'python': sys.version.split()[0], 'model': model_name, 'usage': res.usage,
                         'max_output_tokens': 32768, 'screenshot_errors': len(shot_errors),
                         'screenshot_error_details': shot_errors, 'screenshot_time_ms': int(shot_time[0] * 1000),
                         'retries': 0, 'harness': 'browser-use-pydantic'},
        }
    except Exception as e:  # noqa: BLE001
        envelope['error'] = f'{type(e).__name__}: {e}'
        envelope['metadata']['failure_class'] = 'harness-or-provider'
        traceback.print_exc()
    finally:
        if session is not None:
            await session.repl.close()
        try:
            from browser_harness import _ipc as ipc

            c, token = ipc.connect(os.environ.get('BU_NAME', ''), timeout=2)
            ipc.request(c, token, {'meta': 'shutdown'})
        except Exception:  # noqa: BLE001
            pass
        if browser and browser.get('id'):
            try:
                cloud(f'/browsers/{browser["id"]}', 'PATCH', {'action': 'stop'})
            except Exception as e:  # noqa: BLE001
                cleanup_errors.append(f'Cloud: {e}')
        if cleanup_errors:
            envelope['metadata']['cleanup_errors'] = cleanup_errors
        audit = workspace / 'agent_outputs' / '.browser-use'
        if audit.is_dir():  # Actions skips dotfiles; ship REPL cells/logs as one visible archive
            try:
                with tarfile.open(workspace / 'sdk-audit.tar.gz', 'w:gz') as tar:
                    tar.add(audit, arcname='.browser-use')
                envelope['artifacts'].append('sdk-audit.tar.gz')
            except Exception as e:  # noqa: BLE001
                envelope['metadata']['sdk_audit_archive_error'] = str(e)
        result_path.write_text(json.dumps(envelope, indent=2, default=str) + '\n')
        if span_cm is not None:
            try:
                Laminar.set_span_output(envelope)
                span_cm.__exit__(None, None, None)
                Laminar.flush()
                Laminar.shutdown()
            except Exception:  # noqa: BLE001
                pass
    return 1 if envelope['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
