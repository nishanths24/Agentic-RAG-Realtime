# Agent

Router v1 is rule-based (keywords for time, weather, currency, calculator, policy terms). It is evaluated on `eval/router_questions.jsonl`.

The orchestrator:

- runs at most `MAX_AGENT_STEPS`
- skips duplicate `(tool, args)` pairs
- validates every tool argument in code
- may take one follow-up currency conversion when retrieved policy text contains an amount
- always returns a trace (`trace_id` + steps)

LLM routing is not the default. Enable an OpenAI-compatible client only when it measurably beats the rule router on the labeled set.
