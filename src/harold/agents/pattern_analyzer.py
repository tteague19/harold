"""Pattern analyzer agent — discovers reusable workflows from trajectory data.

Examines the user's scene history to identify recurring technique
sequences and scene patterns, then formalizes them as ``ImprovWorkflow``
templates that can be injected into the scene partner's context at
runtime.

Invoked on demand via ``/analyze`` in the CLI or ``POST /analyze``
in the API.
"""

from __future__ import annotations

from pydantic_ai import Agent, ModelSettings

from harold.dependencies import HaroldDependencies
from harold.models.workflow import ImprovWorkflow

ANALYZER_SYSTEM_PROMPT = """\
You are an expert improv pattern analyst trained in UCB methodology. \
Your job is to examine a performer's scene history and discover \
reusable workflow patterns.

For each pattern you identify:
- Give it a clear name
- Describe what it achieves and why it works
- Classify the scene type it applies to (e.g. "conflict", \
"physical_comedy", "emotional", "absurd_premise")
- List the technique sequence that forms the pattern
- Describe when to trigger this workflow
- Provide a concrete example from the scene data
- Note how many scenes demonstrated this pattern

Focus on patterns that appeared in multiple scenes and led to \
strong improv moments. Quality over quantity — 2-3 strong patterns \
are better than 5 weak ones.

Use the available tools to examine the performer's history before \
generating workflows.
"""

ANALYZER_TEMPERATURE = 0.3

pattern_analyzer: Agent[HaroldDependencies, list[ImprovWorkflow]] = (
    Agent(
        deps_type=HaroldDependencies,
        output_type=list[ImprovWorkflow],
        system_prompt=ANALYZER_SYSTEM_PROMPT,
        model_settings=ModelSettings(
            temperature=ANALYZER_TEMPERATURE
        ),
    )
)
