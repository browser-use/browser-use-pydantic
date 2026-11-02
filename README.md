<img src="https://raw.githubusercontent.com/browser-use/browser-use-pydantic/main/docs/banner.webp" alt="A crystal above a sea of clouds" width="100%" />

# Browser Use Pydantic

**Browser Use, built on Pydantic AI. In Python.**

Pydantic AI + a persistent Python REPL + raw CDP. The agent writes Python, controls Chrome, and builds the helpers it needs as it goes.

```text
Your task → Pydantic AI → persistent Python REPL → raw CDP → Chrome
                 ↑                                            │
                 └──────────── AX tree + screenshots ─────────┘
```

The programmability of [Browser Harness](https://github.com/browser-use/browser-harness) in an agent you can put in your app, with cloud browsers, typed results and follow-ups. It scores the same as [Browser Use Pi](https://github.com/browser-use/browser-use-pi) on BU Bench in matched runs.

## Start

```sh
uv add browser-use-pydantic
export OPENAI_API_KEY=...
export BROWSER_USE_API_KEY=...   # for cloud browsers
```

```python
import asyncio
from browser_use_pydantic import Agent, Browser

async def main():
    async with Agent('Find the top story on Hacker News', browser=Browser(use_cloud=True)) as agent:
        result = await agent.run()
        print(result.final_result())

        agent.add_new_task('Summarize the comments')
        print((await agent.run()).final_result())

asyncio.run(main())
```

## Cloud browsers

`Browser(use_cloud=True)` starts a [Browser Use Cloud](https://cloud.browser-use.com) browser with stealth, proxies and CAPTCHA solving. The agent and its files stay on your machine; only the browser is remote.

- **Watch it live.** The live view opens in your browser when the agent starts (`live_view=False` to skip). The link is also on `agent.browser.live_url`. Anyone with it can drive the browser.
- **Stay logged in.** `Browser(use_cloud=True, cloud_profile_id='work')` takes a profile name or id. Log in once through the live view; cookies are saved when the browser stops.
- **Pick a country.** `cloud_proxy_country_code='de'`, or `None` for no proxy.
- **Files both ways.** `upload_file(selector, path)` sends local files to the remote browser. Files the browser downloads land in `workspace/downloads` when the agent closes.
- **Never left running.** The browser stops when the agent closes, and when your process exits. `cloud_timeout` (minutes, default 60) caps it either way.
- **Run many at once.** Every `Agent` gets its own browser:

```python
tasks = ['Price of an iPhone 17 on apple.com', 'Price of a Pixel 11 on store.google.com']

async def ask(task):
    async with Agent(task, browser=Browser(use_cloud=True)) as agent:
        return (await agent.run()).final_result()

print(await asyncio.gather(*map(ask, tasks)))
```

## Options

```python
Agent(task, llm='openai/gpt-6-luna',     # default; or 'anthropic/...', 'openrouter/...', a Pydantic AI model
      browser=Browser(),                 # Browser(): your Chrome · Browser(cdp_url=...) · Browser(use_cloud=True)
      output_model_schema=MyModel,       # typed result in result.structured_output
      reasoning='medium',                # default
      workspace='./work')                # files the agent saves; default: a temp dir

result = await agent.run(max_steps=100, timeout=1700)
result.final_result()   # answer text, or None if unfinished
result.status           # completed | max_steps | timeout | error
result.partial          # last checkpoint(..., partial=True) when unfinished
```

## How it works

```
  ● agent: python  → new_tab(url); screenshot()           one persistent REPL,
  │                                                      browser-harness helpers + raw cdp()
  ● agent: python  → rows = [...]; checkpoint('rows.json', rows, partial=True)
  │                                                      a hung cell is killed, files survive
  ● agent: finish_from_repl('rows')                       delivers the value it already has
  │
  ✓ result.final_result()
```

The agent keeps its own variables between steps, sees the page as screenshots, and saves progress as it goes, so a timeout still returns partial work.

## Develop

```sh
uv run pytest   # real headless Chrome, scripted model, no provider calls
```

`eval/` runs the agent on the Browser Use eval platform with the same contract as Browser Use Pi.
