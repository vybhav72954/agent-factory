"""Out-of-distribution (paraphrased) prompt experiment.

Tests the paper's own hypothesis (manuscript §8.1): an LLM earns its latency
and API cost over the regex baseline on wording the regex was not built for.

The original 18 test prompts reuse the severity vocabulary that both the regex
and the LLM prompt were written around. This package adds a held-out prompt
bank worded to avoid that vocabulary (plus deliberate negation traps), each
prompt written to express a stated design severity, and evaluates each LLM
under two system prompts: the production word-list prompt and a meaning-based
variant.

Workflow (run from the project root):

    python -m research.paraphrase.prompt_bank   # vocabulary + input-guard checks
    python -m research.paraphrase.collect       # LLM calls (costs API quota), resumable
    python -m research.paraphrase.score         # replay on checkpoints, summary
"""
