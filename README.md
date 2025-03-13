# Building an AI Research Assistant with CrewAI

## Workshop Overview

Welcome to this hands-on workshop where you'll build an intelligent research assistant using CrewAI! By the end of this 2-hour session, you'll have created a working application that uses multiple AI agents to perform comprehensive research on any topic.

### Learning Objectives
- Understand the fundamentals of LLMs and AI agents
- Learn how to create and configure AI agents using CrewAI
- Build a multi-agent system for research tasks
- Create a user-friendly interface with Streamlit
- Deploy your application

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python programming
- Anthropic API key (we'll help you set this up)

## Part 1: Getting Started (20 minutes)

### 1.1 Environment Setup
Let's set up your development environment:

```bash
# Create project directory
mkdir research_assistant
cd research_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install crewai streamlit python-dotenv langchain-openai pytest

# Create .env file
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### 1.2 Understanding the Basics
Before we start coding, let's understand the key concepts:

1. **What are LLMs?**
   - Large Language Models are AI systems trained on vast amounts of text
   - They process input text and generate human-like responses
   - Key parameters:
     - **Temperature** (0-1): Controls randomness in responses
       - Lower (0.1-0.3): More focused, deterministic responses
       - Higher (0.7-0.9): More creative, diverse responses
     - **Context Window**: Maximum amount of text the model can process
     - **Tokens**: Pieces of text (words/subwords) that the model processes
   - Common applications:
     - Text generation and completion
     - Question answering
     - Code generation
     - Text summarization
     - Translation
     - Chatbots
   - Working principles:
     - Process input text into tokens
     - Use attention mechanisms to understand relationships
     - Generate responses based on learned patterns
     - Maintain context throughout the conversation

2. **What is CrewAI?**
   - A framework for creating autonomous AI agents
   - Core concepts:
     - **Agents**: AI entities with specific roles and capabilities
       - Each agent has a goal, backstory, and tools
       - Can be thought of as specialized AI workers
     - **Tasks**: Specific jobs or assignments for agents
       - Contains description and expected output
       - Can be assigned to one or more agents
     - **Crew**: A group of agents working together
       - Manages workflow between agents
       - Coordinates task execution
     - **Tools**: Functions or capabilities agents can use
       - Examples: web search, file reading, API calls
       - Helps agents perform specific actions
   - Key benefits:
     - Modular design for complex tasks
     - Specialized agents for different roles
     - Coordinated workflow between agents
     - Easy integration with external tools
   - Common use cases:
     - Research and analysis
     - Content creation
     - Data processing
     - Problem-solving
     - Decision-making

3. **How CrewAI Works with LLMs**
   - Architecture:
     ```
     User Input → CrewAI → Multiple LLM Agents → Coordinated Response
     ```
   - Workflow:
     1. User provides input/task
     2. CrewAI breaks down the task
     3. Agents process their specific parts
     4. Results are combined and refined
     5. Final output is delivered
   - Key advantages:
     - Parallel processing of tasks
     - Specialized expertise per agent
     - Coordinated problem-solving
     - Scalable architecture

4. **Best Practices for LLM and CrewAI**
   - Prompt Engineering:
     - Clear, specific instructions
     - Context-aware prompts
     - Proper formatting and structure
   - Agent Configuration:
     - Well-defined roles and goals
     - Appropriate backstories
     - Relevant tools and capabilities
   - Task Management:
     - Break down complex tasks
     - Clear expected outputs
     - Logical task sequencing
   - Performance Optimization:
     - Appropriate temperature settings
     - Efficient context usage
     - Proper error handling

5. **Understanding LLM Data Model**
   Here's how data flows through a typical LLM system:

   ```
   ┌─────────────────────────────────────────────────────────────────┐
   │                      Input Processing                            │
   │                                                                 │
   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
   │  │   Text       │    │   Token      │    │   Embedding  │      │
   │  │  Input       │    │  Generation  │    │  Creation    │      │
   │  └──────────────┘    └──────────────┘    └──────────────┘      │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                      Model Architecture                          │
   │                                                                 │
   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
   │  │   Attention  │    │   Feed       │    │   Output     │      │
   │  │   Mechanism  │    │   Forward    │    │   Layer      │      │
   │  └──────────────┘    └──────────────┘    └──────────────┘      │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                      Output Processing                          │
   │                                                                 │
   │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
   │  │   Token      │    │   Text       │    │   Post-      │      │
   │  │   Decoding   │    │   Generation │    │   Processing │      │
   │  └──────────────┘    └──────────────┘    └──────────────┘      │
   │                                                                 │
   └─────────────────────────────────────────────────────────────────┘
   ```

   Key Components:

   1. **Input Processing Layer**
      - Text Input: Raw text from user or system
      - Token Generation: Converts text to tokens (subwords)
      - Embedding Creation: Converts tokens to vector representations

   2. **Model Architecture Layer**
      - Attention Mechanism: Processes relationships between tokens
      - Feed Forward: Neural network processing
      - Output Layer: Generates next token probabilities

   3. **Output Processing Layer**
      - Token Decoding: Converts model output to tokens
      - Text Generation: Combines tokens into coherent text
      - Post-Processing: Formats and refines the output

   Data Flow:
   1. Input text is tokenized and embedded
   2. Embeddings are processed through attention layers
   3. Model generates probability distribution for next token
   4. Tokens are decoded back to text
   5. Output is post-processed and formatted

   Key Concepts:
   - **Tokens**: Subword units (e.g., "unfortunate" → "un" + "fortunate")
   - **Embeddings**: Vector representations of tokens
   - **Attention**: Mechanism for understanding relationships between tokens
   - **Context Window**: Maximum number of tokens the model can process
   - **Temperature**: Controls randomness in token selection
   - **Top-k/Top-p**: Sampling strategies for token selection

## Part 2: Building the Core (30 minutes)

### 2.1 Creating Our Agents
Let's create our first file, `agents.py`:

```python
from crewai import Agent
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="claude-3-sonnet-20240229",  # Using Claude 3.7 Sonnet
    temperature=0.7,
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Create the Research Coordinator agent
research_coordinator = Agent(
    role="Research Coordinator",
    goal="Plan and coordinate research tasks effectively",
    backstory="""You are an experienced research coordinator with expertise in 
    planning and managing research projects. You excel at breaking down complex 
    topics into manageable tasks.""",
    llm=llm,
    verbose=True
)

# Create the Literature Searcher agent
literature_searcher = Agent(
    role="Literature Searcher",
    goal="Find relevant and reliable sources of information",
    backstory="""You are a skilled researcher with expertise in finding and 
    evaluating academic and professional sources. You have a keen eye for 
    credible information.""",
    llm=llm,
    verbose=True
)

# Create the Information Analyst agent
information_analyst = Agent(
    role="Information Analyst",
    goal="Analyze and synthesize information effectively",
    backstory="""You are an expert analyst who excels at understanding complex 
    information and identifying key insights. You can spot patterns and draw 
    meaningful conclusions.""",
    llm=llm,
    verbose=True
)

# Create the Content Writer agent
content_writer = Agent(
    role="Content Writer",
    goal="Create clear and engaging research reports",
    backstory="""You are a professional writer with expertise in creating 
    well-structured and engaging research reports. You excel at presenting 
    complex information in a clear and accessible way.""",
    llm=llm,
    verbose=True
)
```

### 2.2 Defining Tasks
Create `tasks.py`:

```python
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
```

### 2.3 Assembling the Crew
Create `crew.py`:

```python
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
```

## Part 3: Creating the User Interface (20 minutes)

### 3.1 Building with Streamlit
Create `app.py`:

```python
import streamlit as st
from crew import create_research_crew
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide"
)

# Header section
st.title("🤖 AI Research Assistant")
st.markdown("""
    This AI-powered research assistant can help you gather and analyze information on any topic.
    Simply enter your research topic below and let our AI crew do the work for you!
    """)

# Main content
with st.container():
    topic = st.text_input("Enter your research topic:", 
                         placeholder="e.g., Impact of AI on Healthcare")
    
    if st.button("Start Research", type="primary") and topic:
        with st.spinner("Research in progress..."):
        try:
            crew = create_research_crew(topic)
            result = crew.kickoff()
            
            st.markdown("## Research Results")
            st.markdown(result)
            
            st.download_button(
                label="Download Report",
                data=result,
                file_name=f"research_report_{topic.lower().replace(' ', '_')}.md",
                mime="text/markdown"
            )
            
        except Exception as e:
                st.error(f"An error occurred: {str(e)}")
```

## Part 4: Hands-on Activities (15 minutes)

### Activity 1: Custom Agent Creation
Create a new agent for fact-checking:

```python
fact_checker = Agent(
    role="Fact Checker",
    goal="Verify information accuracy and credibility",
    backstory="""You are a meticulous fact-checker with expertise in verifying 
    information from various sources. You excel at identifying reliable data 
    and spotting inconsistencies.""",
    llm=llm,
    verbose=True
)
```

### Activity 2: Adding a Custom Tool
Create a web scraping tool:

```python
from bs4 import BeautifulSoup
import requests

def web_scraper(url):
    """Scrape content from a webpage"""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

# Add to agent tools
agent.tools.append(web_scraper)
```

## Part 5: Testing and Deployment (15 minutes)

### 5.1 Running Tests
Create `test_crew.py`:

```python
import pytest
from crew import create_research_crew

def test_crew_creation():
    crew = create_research_crew("Test Topic")
    assert crew is not None
    assert len(crew.agents) == 4
    assert len(crew.tasks) == 4
```

Run the tests:
```bash
pytest
```

### 5.2 Deploying Your App
1. Run locally:
```bash
streamlit run app.py
```

2. Deploy to Streamlit Cloud:
   - Push to GitHub
   - Connect at share.streamlit.io
   - Deploy!

## Troubleshooting Tips

If you encounter issues:

1. **API Key Problems**
   - Check your .env file for ANTHROPIC_API_KEY
   - Verify the key is valid
   - Ensure it's being loaded correctly

2. **Environment Issues**
   - Verify Python version
   - Check package installations
   - Ensure virtual environment is activated

3. **Performance Optimization**
   - Adjust temperature settings
   - Implement caching
   - Use appropriate model sizes

## Next Steps

1. **Extend Your Assistant**
   - Add more specialized agents
   - Implement custom tools
   - Enhance the UI

2. **Learn More**
   - Explore CrewAI documentation
   - Try different agent configurations
   - Experiment with various LLM models

3. **Share Your Work**
   - Deploy to Streamlit Cloud
   - Share with others
   - Get feedback and improve

## Resources

- [CrewAI Documentation](https://docs.crewai.com)
- [Streamlit Documentation](https://docs.streamlit.io)
- [LangChain Documentation](https://python.langchain.com)
- [OpenAI API Documentation](https://platform.openai.com/docs)