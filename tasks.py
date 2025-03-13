from crewai import Task

def create_research_tasks(topic, research_coordinator, literature_searcher, 
                         information_analyst, content_writer):
    # Task 1: Research Planning
    planning_task = Task(
        description=f"""Create a detailed research plan for the topic: {topic}
        Include:
        1. Key aspects to investigate
        2. Types of sources to look for
        3. Analysis approach
        4. Report structure""",
        expected_output="A comprehensive research plan with clear steps and structure",
        agent=research_coordinator
    )

    # Task 2: Literature Search
    search_task = Task(
        description=f"""Find relevant sources and information about: {topic}
        Focus on:
        1. Academic sources
        2. Professional publications
        3. Recent developments
        4. Key statistics and data""",
        expected_output="A collection of relevant sources and key information",
        agent=literature_searcher
    )

    # Task 3: Information Analysis
    analysis_task = Task(
        description=f"""Analyze the information gathered about: {topic}
        Include:
        1. Key findings
        2. Patterns and trends
        3. Supporting evidence
        4. Potential implications""",
        expected_output="A detailed analysis of the gathered information",
        agent=information_analyst
    )

    # Task 4: Report Writing
    writing_task = Task(
        description=f"""Create a comprehensive research report about: {topic}
        Structure the report with:
        1. Executive summary
        2. Key findings
        3. Analysis
        4. Conclusions
        5. Recommendations""",
        expected_output="A well-structured research report",
        agent=content_writer
    )

    return [planning_task, search_task, analysis_task, writing_task] 