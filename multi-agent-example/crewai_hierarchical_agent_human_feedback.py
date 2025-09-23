from langchain_openai import AzureChatOpenAI
from crewai import Agent, Process
from crewai import Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai_tools import YoutubeVideoSearchTool
from crewai import Crew
from crewai import LLM
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'

GEMINI_API_KEY = ""
os.environ["SERPER_API_KEY"] = ''
my_azure_api_key = ""
AZURE_API_KEY = my_azure_api_key
AZURE_API_BASE = "https://ammondal-llm-test.openai.azure.com/"
AZURE_API_VERSION = "2025-04-01-preview"

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["OPENAI_API_KEY"] = my_azure_api_key


class MyCrewAgent:

    def __init__(self):
        print(f"Initializing CrewAI agents..")
        # Create a search tool
        # Method # 1
        # General search across Youtube content without specifying a video URL, so the agent can search within any Youtube video content it learns about irs url during its operation
        #self.search_tool = YoutubeVideoSearchTool()

        # Method # 2
        # Targeted search within a specific Youtube video's content
        self.search_tool = YoutubeVideoSearchTool('https://www.youtube.com/watch?v=R0ds4Mwhy-8', add_video_info=False)
       
        self.init()
        # Define the hierarchical crew with a management LLM
        self.research_crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[self.research_task, self.writing_task],
            verbose=True,
            memory=True
        )
        print(f"CrewAI agents initialized.")


    def init(self):
        self.prepare_all_agents()
        self.prepare_all_tasks()            

    def prepare_all_agents(self):
        print(f"Initializing research analyst agent...") 
        # Initialize the Gemini model using ChatGoogleGenerativeAI
        #self.gemini_llm = LLM(
        #    model="gemini/gemini-2.0-flash",
        #    temperature=0.7,
        #    verbose=True,
        #    api_key=GEMINI_API_KEY
        #)

        # Initialize the Azure OpenAI model using AzureChatOpenAI
        self.azure_openai_llm = LLM(
            model="azure/gpt-4.1",
            api_version="2023-05-15",
            api_base=AZURE_API_BASE,
            api_key=AZURE_API_KEY
        )

        # Define the research agent
        self.researcher = Agent(
            role='Video Content Researcher',
            goal='Extract key insights from YouTube videos',
            backstory=(
                "You are a skilled researcher who excels at extracting valuable insights from video content. "
                "You focus on gathering accurate and relevant information from YouTube to support your team."
            ),
            verbose=True,
            tools=[self.search_tool],
            memory = True
        )

        # Define the writing agent
        self.writer = Agent(
            role='Tech Article Writer',
            goal='Craft an article based on the research insights',
            backstory=(
                "You are an experienced writer known for turning complex information into engaging and accessible articles. "
                "Your work helps make advanced technology topics understandable to a broad audience."
            ),
            verbose=True,
            tools=[self.search_tool],  # The writer may also use the YouTube tool for additional context 
            memory = True
        )
        print(f"All agents initialized.")

    def prepare_all_tasks(self):
        print(f"Initializing tasks for all agents...")
        # Create the research task
        self.research_task = Task(
            description=(
                "Research and extract key insights from the given YouTube video regarding Educative. "
                "Compile your findings in a detailed summary."
            ),
            expected_output='A summary of the key insights from the YouTube video',
            agent=self.azure_openai_llm
        )

        # Create the writing task
        self.writing_task = Task(
            description=(
                "Using the summary provided by the researcher, write a compelling article on what is Educative. "
                "Ensure the article is well-structured and engaging for a tech-savvy audience."
            ),
            expected_output='A well-written article on Educative based on the YouTube video research.',
            agent=self.azure_openai_llm,
            human_input=True  # Allow for human feedback after the draft
        )
        print(f"All tasks for agents initialized.")
    

    def run_event_planning_crew(self):
        print(f"Running CrewAI agents...")
        try:
            result = self.research_crew.kickoff()
            print("Crew kickoff result:\n", result)
        except Exception as e:
            print(f"CrewAI kickoff failed: {e}")
            result = None

        print(f"CrewAI agents run completed.")


MyCrewAgent().run_event_planning_crew()
        

