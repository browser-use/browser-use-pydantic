# browser-use-pydantic

Browser Use on Pydantic AI. The agent writes Python against raw CDP ([browser-harness](https://github.com/browser-use/browser-harness)).

```sh
uv add browser-use-pydantic
export OPENAI_API_KEY=...
```

```python
import asyncio
from browser_use_pydantic import Agent, Browser

async def main():
    async with Agent('Find the top story on Hacker News', llm='openai/gpt-5.6-luna') as agent:
        result = await agent.run()
        print(result.final_result())

asyncio.run(main())
```

## Options

```python
Agent(
    task,
    llm='openai/gpt-5.6-luna',     # or 'anthropic/...', 'openrouter/...', or a Pydantic AI model
    browser=Browser(),             # see below
    output_model_schema=MyModel,   # typed result in result.structured_output
    reasoning='medium',
    workspace='./work',            # files the agent saves; default: a temp dir
)

result = await agent.run(max_steps=100, timeout=1700)
result.final_result()      # answer text, or None if it did not finish
result.structured_output   # MyModel instance
result.status              # completed | max_steps | timeout | error
result.partial             # last checkpoint(..., partial=True) when unfinished

agent.add_new_task('Now summarize the comments')  # next run() continues the conversation
await agent.run()
```

## Browser

```python
Browser()                                   # your local Chrome
Browser(cdp_url='http://localhost:9222')    # any browser with remote debugging
Browser(use_cloud=True)                     # Browser Use Cloud, needs BROWSER_USE_API_KEY
```

## Develop

`uv run pytest` runs a real headless Chrome with a scripted model; it makes no provider calls. `eval/` is the browser-use-pi eval contract.
