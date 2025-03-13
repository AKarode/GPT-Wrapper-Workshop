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