"""browser-use style API: Agent(task, llm, browser).run() -> result.final_result()."""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import uuid
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

os.environ.setdefault('PYDANTIC_AI_NO_BANNER', '1')  # a library should not print an ad on every run

from .agent import Repl, RunResult, Session, build_agent, model_ref, run


CLOUD_API = 'https://api.browser-use.com/api/v4'
CLOUD_PROMPT = (
    'The browser is a remote Browser Use Cloud browser. upload_file(selector, path) takes local workspace paths. '
    'Files the browser downloads are copied to downloads/ in the workspace when the session closes; do not change '
    'the browser download directory.'
)


def _profile_id(profile: str) -> str:
    """A cloud profile id, or the id of the one profile with exactly this name."""
    if re.fullmatch(r'[0-9a-f-]{36}', profile):
        return profile
    items = _cloud(f'/profiles?pageSize=100&query={urllib.parse.quote(profile)}').get('items', [])
    ids = [p['id'] for p in items if p.get('name') == profile]
    if len(ids) != 1:
        raise ValueError(f'{len(ids)} cloud profiles are named {profile!r}; pass the profile id')
    return ids[0]


def _cloud(path: str, method: str = 'GET', body: dict | None = None) -> dict:
    from browser_harness.auth import get_browser_use_api_key

    req = urllib.request.Request(f'{CLOUD_API}{path}', method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={'X-Browser-Use-API-Key': get_browser_use_api_key(),
                                          'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors='replace')[:500]
        hint = (' Create a key with v4 browser access at https://cloud.browser-use.com/settings?tab=api-keys and set '
                'BROWSER_USE_API_KEY.') if 'missing required scope' in detail else ''
        raise RuntimeError(f'Browser Use Cloud {method} {path} failed: HTTP {e.code} {detail}.{hint}') from None


class Browser:
    """Which browser the agent drives.

    Browser()                 your local Chrome (browser-harness default daemon)
    Browser(cdp_url=...)      any running browser with remote debugging
    Browser(use_cloud=True)   a Browser Use Cloud browser: live view, profiles, proxies; stopped on close
    """

    def __init__(self, *, cdp_url: str | None = None, use_cloud: bool = False,
                 cloud_profile_id: str | None = None, cloud_proxy_country_code: str | None = 'us',
                 cloud_timeout: int = 60, live_view: bool = True):
        """cloud_profile_id: a cloud profile id or name (logins persist in it). cloud_proxy_country_code=None turns
        the proxy off. cloud_timeout: minutes (1-240). live_view: print and open the live view URL."""
        if cdp_url and use_cloud:
            raise ValueError('Pass cdp_url or use_cloud, not both')
        self.cdp_url, self.use_cloud, self.live_view = cdp_url, use_cloud, live_view
        self.profile, self.proxy, self.timeout = cloud_profile_id, cloud_proxy_country_code, cloud_timeout
        self.name: str | None = None  # browser-harness daemon this agent owns; None = the shared default daemon
        self.id: str | None = None  # cloud browser id
        self.live_url: str | None = None  # interactive; anyone with the link can drive the browser
        self.stopped = False

    async def start(self) -> dict[str, str]:
        """Attach a browser-harness daemon; returns the env the REPL worker needs."""
        from browser_harness import admin

        if not self.cdp_url and not self.use_cloud:
            await asyncio.to_thread(admin.ensure_daemon)
            return {}
        self.name = f'bup{uuid.uuid4().hex[:8]}'
        if not self.use_cloud:
            ws = self.cdp_url
            if ws.startswith('http'):
                ws = (await asyncio.to_thread(
                    lambda: json.load(urllib.request.urlopen(f'{self.cdp_url.rstrip("/")}/json/version', timeout=30))
                ))['webSocketDebuggerUrl']
            await asyncio.to_thread(admin.ensure_daemon, name=self.name, env={'BU_CDP_WS': ws, 'BH_TAB_MARKER': '0'})
            return {'BU_NAME': self.name}
        body = {'proxyCountryCode': self.proxy, 'timeout': self.timeout,
                'browserScreenWidth': 1440, 'browserScreenHeight': 900}  # the evaluated cloud setup
        if self.profile:
            body['profileId'] = await asyncio.to_thread(_profile_id, self.profile)
        info = await asyncio.to_thread(_cloud, '/browsers', 'POST', body)
        self.id, self.live_url = info['id'], info.get('liveUrl')
        atexit.register(self._stop_now)  # never leave a billed browser behind when the process exits
        try:
            ws = (await asyncio.to_thread(
                lambda: json.load(urllib.request.urlopen(f'{info["cdpUrl"]}/json/version', timeout=30))
            ))['webSocketDebuggerUrl']
            await asyncio.to_thread(admin.ensure_daemon, name=self.name, env={'BU_CDP_WS': ws, 'BH_TAB_MARKER': '0'})
        except BaseException:
            await self.stop()
            raise
        if self.live_view and self.live_url:
            print(f'Live view: {self.live_url}')
            webbrowser.open(self.live_url, new=2)
        return {'BU_NAME': self.name, 'BU_REMOTE_BROWSER': '1'}

    def _stop_now(self) -> None:
        """Stop our daemon, then the cloud browser (PATCH stop ends billing and saves the profile)."""
        from browser_harness import admin

        name, self.name = self.name, None
        if name:
            try:
                admin.restart_daemon(name)
            except Exception:  # noqa: BLE001 - cleanup must not mask the result
                pass
        if self.use_cloud and self.id and not self.stopped:
            for attempt in range(3):
                try:
                    _cloud(f'/browsers/{self.id}', 'PATCH', {'action': 'stop'})
                    self.stopped = True
                    return
                except Exception as e:  # noqa: BLE001
                    error = e
                    time.sleep(0.5 * (attempt + 1))
            logging.getLogger(__name__).warning('Could not stop cloud browser %s: %s', self.id, error)

    async def stop(self) -> None:
        """Stop the daemon this agent started (and its cloud browser). Never touches your local Chrome."""
        await asyncio.to_thread(self._stop_now)
        if self.use_cloud:
            atexit.unregister(self._stop_now)

    async def download_files(self, dest: str | Path) -> list[Path]:
        """Copy the files the cloud browser downloaded into dest. Returns the local paths."""
        if not self.id:
            return []
        dest, out, cursor = Path(dest), [], None
        while True:
            q = f'/browsers/{self.id}/downloads?includeUrls=true&limit=100' + (f'&cursor={cursor}' if cursor else '')
            page = await asyncio.to_thread(_cloud, q)
            for f in page.get('files', []):
                path = dest / Path(f['path']).name
                path.parent.mkdir(parents=True, exist_ok=True)
                data = await asyncio.to_thread(lambda u=f['url']: urllib.request.urlopen(u, timeout=120).read())
                path.write_bytes(data)
                out.append(path)
            cursor = page.get('nextCursor')
            if not page.get('hasMore') or not cursor:
                return out


class Agent:
    """agent = Agent(task, llm='openai/gpt-6-luna'); result = await agent.run(); result.final_result()"""

    def __init__(self, task: str, llm: Any = 'openai/gpt-6-luna', *, browser: Browser | None = None,
                 output_model_schema: Any = str, reasoning: str = 'medium', workspace: str | Path | None = None):
        self.task, self.llm, self.browser = task, model_ref(llm), browser or Browser()
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix='browser-use-')).resolve()
        self._agent = build_agent(self.llm, output_type=output_model_schema, reasoning_effort=reasoning,
                                  instructions=CLOUD_PROMPT if self.browser.use_cloud else '')
        self._repl: Repl | None = None
        self._history: list[ModelMessage] | None = None
        self._continue = False

    def add_new_task(self, task: str) -> None:
        """The next run() continues this conversation with a new task."""
        self.task, self._continue = task, True

    async def run(self, max_steps: int = 100, timeout: float = 1700) -> RunResult:
        if self._repl is None:
            self.workspace.mkdir(parents=True, exist_ok=True)
            self._repl = Repl(self.workspace, await self.browser.start())
        (self._repl.state / 'partial.json').unlink(missing_ok=True)  # a partial belongs to one run
        session = Session(repl=self._repl, timeout_s=timeout, max_steps=max_steps)
        name = self.llm if isinstance(self.llm, str) else getattr(self.llm, 'model_name', '')
        result = await run(self._agent, self.task, session, name,
                           message_history=self._history if self._continue else None)
        self._history, self._continue = result.messages, False
        return result

    async def close(self) -> None:
        """Stop the REPL and the browser; a cloud browser's downloads land in workspace/downloads."""
        if self._repl:
            await self._repl.close()
            self._repl = None
        await self.browser.stop()
        try:
            await self.browser.download_files(self.workspace / 'downloads')
        except Exception as e:  # noqa: BLE001 - the result is already delivered
            logging.getLogger(__name__).warning('Could not fetch cloud browser downloads: %s', e)

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
