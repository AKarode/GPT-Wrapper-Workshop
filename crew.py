from crewai import Crew
from agents import (research_coordinator, literature_searcher, 
                   information_analyst, content_writer)
from tasks import create_research_tasks

def create_research_crew(topic):
    tasks = create_research_tasks(
        topic,
        research_coordinator,
        literature_searcher,
        information_analyst,
        content_writer
    )

    crew = Crew(
        agents=[
            research_coordinator,
            literature_searcher,
            information_analyst,
            content_writer
        ],
        tasks=tasks,
        verbose=True
    )

    return crew 