from langchain_openai import AzureChatOpenAI
from crewai import Agent, Process
from crewai import Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
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
        self.search_tool = SerperDevTool()
        self.scrape_tool = ScrapeWebsiteTool()
        self.init()
        # Define the hierarchical crew with a management LLM
        self.research_crew = Crew(
            agents=[self.research_analyst_agent, self.report_writer_agent, self.report_editor_agent],
            tasks=[self.data_collection_task, self.data_analysis_task, self.report_writing_task, self.report_assessment_task],
            manager_llm=self.azure_openai_llm,
            process=Process.hierarchical,
            verbose=True
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
        
        # Research Analyst Agent using SerperSearch
        self.research_analyst_agent = Agent(
            role="Research Analyst",
            goal="Create and analyze research points to provide comprehensive insights on various topics.",
            backstory="Specializing in research analysis, this agent employs advanced methodologies to generate detailed research points and insights. With a deep understanding of research frameworks and a talent for synthesizing information, the Research Analyst Agent is instrumental in delivering thorough and actionable research outcomes.",
            llm=self.azure_openai_llm,
            verbose=True,
            allow_delegation=True,
            tools=[self.scrape_tool, self.search_tool]
        )

        # Report Writer Agent using Azure OpenAI
        self.report_writer_agent = Agent(
            role="Report Writer",
            goal="Compile the analyzed data into a comprehensive and well-structured research report.",
            backstory="You are skilled at transforming complex information into clear, concise, and informative reports.",
            llm=self.azure_openai_llm,
            verbose=True,
            allow_delegation=True
        )

        # Report Editor Agent using Azure OpenAI
        self.report_editor_agent = Agent(
            role="Report Editor",
            goal="Review and refine research reports to ensure clarity, accuracy, and adherence to standards.",
            backstory="With a keen eye for detail and a strong background in report editing, this agent ensures that research reports are polished, coherent, and meet high-quality standards. Skilled in revising content for clarity and consistency, the Report Editor Agent plays a critical role in finalizing research outputs.",
            llm=self.azure_openai_llm,
            verbose=True
        ) 
        print(f"All agents initialized.")

    def prepare_all_tasks(self):

        print(f"Initializing tasks for all agents...")
        
        self.data_collection_task = Task(
            description=(
                "Collect data from relevant sources about the given {topic}."
                "Focus on identifying key trends, benefits, and challenges."
            ),
            expected_output=(
                "A comprehensive dataset that includes recent studies, statistics, and expert opinions."
            ),
            agent=self.research_analyst_agent,
        )

        self.data_analysis_task = Task(
            description=(
                "Analyze the collected data to identify key trends, benefits, and challenges for the {topic}."
            ),
            expected_output=(
                "A detailed analysis report highlighting the most significant findings."
            ),
            agent=self.research_analyst_agent,
        )

        self.report_writing_task = Task(
            description=(
                "Write a comprehensive research report that clearly presents the findings from the data analysis report"
            ),
            expected_output=(
                "A well-structured research report that provides insights about the topic."
            ),
            agent=self.report_writer_agent,
        )

        self.report_assessment_task = Task(
            description=(
                "Review and rewrite the research report to ensure clarity, accuracy, and adherence to standards."
            ),
            expected_output=(
                "A polished, coherent research report that meets high-quality standards and effectively communicates the findings."
            ),
            agent=self.report_editor_agent,
        )
        print(f"All tasks for agents initialized.")
    

    def run_event_planning_crew(self):
        print(f"Running CrewAI agents...")
        # Define the input for the research topic
        research_inputs = {
            'topic': 'The impact of AI on modern healthcare systems'
        }
        try:
            result = self.research_crew.kickoff(inputs=research_inputs)
            print("Crew kickoff result:\n", result)
        except Exception as e:
            print(f"CrewAI kickoff failed: {e}")
            result = None

        print(f"CrewAI agents run completed.")


MyCrewAgent().run_event_planning_crew()
        

