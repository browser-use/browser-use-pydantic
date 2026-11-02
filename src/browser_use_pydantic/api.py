"""browser-use style API: Agent(task, llm, browser).run() -> result.final_result()."""

from __future__ import annotations

import asyncio
import json
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage

from .agent import Repl, RunResult, Session, build_agent, model_ref, run


class Browser:
    """Which browser the agent drives.

    Browser()                 your local Chrome (browser-harness default daemon)
    Browser(cdp_url=...)      any running browser with remote debugging
    Browser(use_cloud=True)   a Browser Use Cloud browser (BROWSER_USE_API_KEY); stopped on close
    """

    def __init__(self, *, cdp_url: str | None = None, use_cloud: bool = False,
                 cloud_profile_id: str | None = None, cloud_proxy_country_code: str | None = 'us',
                 cloud_timeout: int | None = None):
        if cdp_url and use_cloud:
            raise ValueError('Pass cdp_url or use_cloud, not both')
        self.cdp_url, self.use_cloud = cdp_url, use_cloud
        self.cloud = {'profileId': cloud_profile_id, 'proxyCountryCode': cloud_proxy_country_code,
                      'timeout': cloud_timeout}
        self.name: str | None = None  # browser-harness daemon this agent owns; None = the shared default daemon

    async def start(self) -> dict[str, str]:
        """Attach a browser-harness daemon; returns the env the REPL worker needs."""
        from browser_harness import admin

        if not self.cdp_url and not self.use_cloud:
            await asyncio.to_thread(admin.ensure_daemon)
            return {}
        self.name = f'bup{uuid.uuid4().hex[:8]}'
        if self.use_cloud:
            kwargs = {k: v for k, v in self.cloud.items() if v is not None or k == 'proxyCountryCode'}
            await asyncio.to_thread(admin.start_remote_daemon, self.name, **kwargs)
        else:
            ws = self.cdp_url
            if ws.startswith('http'):
                ws = await asyncio.to_thread(
                    lambda: json.load(urllib.request.urlopen(f'{self.cdp_url.rstrip("/")}/json/version', timeout=30))
                )
                ws = ws['webSocketDebuggerUrl']
            await asyncio.to_thread(admin.ensure_daemon, name=self.name, env={'BU_CDP_WS': ws})
        return {'BU_NAME': self.name}

    async def stop(self) -> None:
        """Stop the daemon this agent started (and its cloud browser). Never touches your local Chrome."""
        if not self.name:
            return
        from browser_harness import admin

        name, self.name = self.name, None
        try:
            await asyncio.to_thread(admin.stop_remote_daemon if self.use_cloud else admin.restart_daemon, name)
        except Exception:  # noqa: BLE001 - cleanup must not mask the result
            pass


class Agent:
    """agent = Agent(task, llm='openai/gpt-5.6-luna'); result = await agent.run(); result.final_result()"""

    def __init__(self, task: str, llm: Any = 'openai/gpt-5.6-luna', *, browser: Browser | None = None,
                 output_model_schema: Any = str, reasoning: str = 'medium', workspace: str | Path | None = None):
        self.task, self.llm, self.browser = task, model_ref(llm), browser or Browser()
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix='browser-use-')).resolve()
        self._agent = build_agent(self.llm, output_type=output_model_schema, reasoning_effort=reasoning)
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
        if self._repl:
            await self._repl.close()
            self._repl = None
        await self.browser.stop()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
