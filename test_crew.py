import pytest
from crew import create_research_crew

def test_crew_creation():
    crew = create_research_crew("Test Topic")
    assert crew is not None
    assert len(crew.agents) == 4
    assert len(crew.tasks) == 4 