"""Agent definitions with tool registrations.

Imports all tool modules to trigger ``@agent.tool`` decorator
registration on agent instances. Consumer modules should import
agents from this package rather than from individual agent modules.
"""

import harold.tools.analysis_tools as analysis_tools  # noqa: F401
import harold.tools.coaching_tools as coaching_tools  # noqa: F401
import harold.tools.memory_tools as memory_tools  # noqa: F401
import harold.tools.scene_tools as scene_tools  # noqa: F401
from harold.agents.coach import coach
from harold.agents.pattern_analyzer import pattern_analyzer
from harold.agents.scene_partner import scene_partner

__all__ = ["coach", "pattern_analyzer", "scene_partner"]
