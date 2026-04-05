"""Tests for workflow model validation and in-memory workflow storage.

Verifies ImprovWorkflow construction, storage, keyword-based
retrieval, and listing from the in-memory trajectory backend.
"""

from __future__ import annotations

import pytest

from harold.memory.backends.in_memory import InMemoryTrajectoryMemory
from harold.models.workflow import ImprovWorkflow

WORKFLOW_NAME = "conflict-escalation"
WORKFLOW_DESCRIPTION = "Escalate a conflict through three beats"
WORKFLOW_SCENE_TYPE = "conflict"
WORKFLOW_TECHNIQUES = ["yes-and", "heightening", "emotional-truth"]
WORKFLOW_TRIGGER = "When two characters disagree about something"
WORKFLOW_EXAMPLE = "Two chefs argued over soup seasoning"
WORKFLOW_SUCCESS_COUNT = 3


@pytest.fixture
def sample_workflow() -> ImprovWorkflow:
    """Provide a sample ImprovWorkflow for testing.

    Returns:
        An ImprovWorkflow with deterministic test values.
    """
    return ImprovWorkflow(
        name=WORKFLOW_NAME,
        description=WORKFLOW_DESCRIPTION,
        scene_type=WORKFLOW_SCENE_TYPE,
        technique_sequence=WORKFLOW_TECHNIQUES,
        trigger_description=WORKFLOW_TRIGGER,
        example_summary=WORKFLOW_EXAMPLE,
        success_count=WORKFLOW_SUCCESS_COUNT,
    )


@pytest.fixture
def memory() -> InMemoryTrajectoryMemory:
    """Provide a fresh empty trajectory memory instance.

    Returns:
        An empty InMemoryTrajectoryMemory.
    """
    return InMemoryTrajectoryMemory()


def test_workflow_model_validates_fields() -> None:
    """Verify that ImprovWorkflow enforces field constraints."""
    workflow = ImprovWorkflow(
        name=WORKFLOW_NAME,
        description=WORKFLOW_DESCRIPTION,
        scene_type=WORKFLOW_SCENE_TYPE,
        technique_sequence=WORKFLOW_TECHNIQUES,
        trigger_description=WORKFLOW_TRIGGER,
        example_summary=WORKFLOW_EXAMPLE,
        success_count=WORKFLOW_SUCCESS_COUNT,
    )
    assert workflow.name == WORKFLOW_NAME
    assert len(workflow.technique_sequence) == len(WORKFLOW_TECHNIQUES)


def test_workflow_model_rejects_empty_name() -> None:
    """Verify that ImprovWorkflow rejects an empty name."""
    with pytest.raises(ValueError):
        ImprovWorkflow(
            name="",
            description=WORKFLOW_DESCRIPTION,
            scene_type=WORKFLOW_SCENE_TYPE,
            technique_sequence=WORKFLOW_TECHNIQUES,
            trigger_description=WORKFLOW_TRIGGER,
            example_summary=WORKFLOW_EXAMPLE,
            success_count=WORKFLOW_SUCCESS_COUNT,
        )


async def test_store_and_retrieve_all_workflows(
    memory: InMemoryTrajectoryMemory,
    sample_workflow: ImprovWorkflow,
) -> None:
    """Verify that a stored workflow can be retrieved via get_all_workflows.

    Args:
        memory: The in-memory trajectory memory fixture.
        sample_workflow: The sample workflow fixture.
    """
    await memory.store_workflow(sample_workflow)

    workflows = await memory.get_all_workflows()
    assert len(workflows) == 1
    assert workflows[0].name == WORKFLOW_NAME


async def test_get_workflows_for_scene_matches_keywords(
    memory: InMemoryTrajectoryMemory,
    sample_workflow: ImprovWorkflow,
) -> None:
    """Verify that workflow retrieval matches on scene description keywords.

    Args:
        memory: The in-memory trajectory memory fixture.
        sample_workflow: The sample workflow fixture.
    """
    await memory.store_workflow(sample_workflow)

    results = await memory.get_workflows_for_scene("a big conflict")
    assert len(results) == 1
    assert results[0].name == WORKFLOW_NAME


async def test_get_workflows_for_scene_no_match(
    memory: InMemoryTrajectoryMemory,
    sample_workflow: ImprovWorkflow,
) -> None:
    """Verify that unrelated scene descriptions return no workflows.

    Args:
        memory: The in-memory trajectory memory fixture.
        sample_workflow: The sample workflow fixture.
    """
    await memory.store_workflow(sample_workflow)

    results = await memory.get_workflows_for_scene("underwater ballet")
    assert results == []


async def test_get_workflows_empty_memory(
    memory: InMemoryTrajectoryMemory,
) -> None:
    """Verify that get_all_workflows returns empty on fresh memory.

    Args:
        memory: The in-memory trajectory memory fixture.
    """
    workflows = await memory.get_all_workflows()
    assert workflows == []
