from langchain_openai import AzureChatOpenAI
from crewai import Agent, Process
from crewai import Task
from crewai_tools import SerperDevTool
from crewai import Crew
from crewai import LLM
import warnings
import os
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

GEMINI_API_KEY = "<<>>"
os.environ["SERPER_API_KEY"] = '<<>>'
my_azure_api_key = "<<>>"
AZURE_API_KEY = my_azure_api_key
AZURE_API_BASE = "https://ammondal-llm-test.openai.azure.com/"
AZURE_API_VERSION = "2025-04-01-preview"

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["OPENAI_API_KEY"] = my_azure_api_key


class MyCrewAgent:

    def __init__(self):
        print(f"Initializing CrewAI agents..")
        # Create a search tool
        self.search_tool = SerperDevTool()
        
        article_researcher, research_task = self.prepare_article_researcher_agent()
        article_writer, writing_task = self.prepare_article_writer_agent()
        self.event_planning_crew = Crew(
            agents=[article_researcher, article_writer],
            tasks=[research_task, writing_task],
            verbose=True,
            memory=True,
            process=Process.sequential
            )
        print(f"CrewAI agents initialized.")
                

    def prepare_article_researcher_agent(self):
        print(f"Initializing gemini llm agent...") 

        # Initialize the Gemini model using ChatGoogleGenerativeAI
        self.gemini_llm = LLM(
            model="gemini/gemini-2.0-flash",
            temperature=0.7,
            verbose=True,
            api_key=GEMINI_API_KEY
        )
        
        # Data Researcher Agent using Gemini and SerperSearch
        self.article_researcher=Agent(
            role="Senior Researcher",
            goal='Unccover ground breaking technologies in {topic}',
            verbose=True,
            memory=True,
            backstory=(
                "Driven by curiosity, you're at the forefront of"
                "innovation, eager to explore and share knowledge that could change"
                "the world."
            ),
            tools=[self.search_tool],
            llm=self.gemini_llm,
            allow_delegation=True
        )

        # Research Task
        self.research_task = Task(
            description=(
                "Conduct a thorough analysis on the given {topic}."
                "Utilize SerperSearch for any necessary online research. "
                "Summarize key findings in a detailed report."
            ),
            expected_output='A detailed report on the data analysis with key insights.',
            tools=[self.search_tool],
            agent=self.article_researcher,
        )
        return self.article_researcher, self.research_task        

    def prepare_article_writer_agent(self):
        print(f"Initializing azure openai llm agent...") 
        # Create the LLM instance
        self.azure_openai_llm = LLM(
            model="azure/gpt-4.1",
            api_version="2023-05-15",
            api_base=AZURE_API_BASE,
            api_key=AZURE_API_KEY
        )

        # Article Writer Agent using Azure OpenAI and SerperSearch
        self.article_writer = Agent(
            role='Writer',
            goal='Narrate compelling tech stories about {topic}',
            verbose=True,
            memory=True,
            backstory=(
                "With a flair for simplifying complex topics, you craft"
                "engaging narratives that captivate and educate, bringing new"
                "discoveries to light in an accessible manner."
            ),
            tools=[self.search_tool],
            llm=self.azure_openai_llm,
            allow_delegation=False
        )
        # Writing Task
        self.writing_task = Task(
            description=(
                "Write an insightful article based on the data analysis report. "
                "The article should be clear, engaging, and easy to understand."
            ),
            expected_output='A 6-paragraph article summarizing the data insights.',
            agent=self.article_writer,
        )
        return self.article_writer, self.writing_task

    

    def run_event_planning_crew(self):
        print(f"Running CrewAI agents...")
        inputs = {
            'topic': 'How to make yellowstone roadtrip interesting with family and toddler ?'
        }

        try:
            result = self.event_planning_crew.kickoff(inputs=inputs)
            print("Crew kickoff result:\n", result)
        except Exception as e:
            print(f"CrewAI kickoff failed: {e}")
            result = None

        print(f"CrewAI agents run completed.")


MyCrewAgent().run_event_planning_crew()
        

