# browser-use-pydantic

Pydantic AI owns the loop. browser-harness owns the browser. One tool: `python`, a persistent REPL with the harness helpers and raw `cdp()`.

```python
from browser_use_pydantic import Repl, Session, build_agent, run

agent = build_agent('openai-responses:gpt-6-luna', reasoning_effort='medium')
session = Session(repl=Repl(workspace))          # BU_NAME selects the browser-harness daemon
result = await run(agent, 'Find ...', session, 'gpt-6-luna')
```

Carried over from browser-use-pi because evals rewarded them: `finish_from_repl` delivers an existing value, screenshots are native images (last 2 kept), OpenAI server-side compaction, one forced delivery turn at 90% of time/steps, `checkpoint(..., partial=True)` survives timeouts, a killable worker so a hung cell never kills the run.

- `src/browser_use_pydantic/agent.py` agent, tool, delivery, budgets
- `src/browser_use_pydantic/worker.py` REPL child process
- `src/browser_use_pydantic/prompt.py` system prompt
- `src/browser_use_pydantic/monty.py` experimental backend (`uv sync --extra monty`): `build_agent(backend='monty', workspace=ws)` + `Session(repl=MontyRepl(ws))` runs code in Pydantic's Monty sandbox via CodeMode `run_code`; eval option `code_backend: "monty"`
- `eval/` the browser-use-pi eval contract (`setup.sh`, `run.mjs` -> `run.py`)

Tests: `uv run pytest` (real headless Chrome, scripted model, no provider calls).
