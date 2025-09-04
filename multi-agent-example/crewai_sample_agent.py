from crewai import Agent
from crewai import Task
from crewai_tools import SerperDevTool
from crewai import Crew


class MyCrewAgent:

    def __init__(self):
        print(f"Initializing CrewAI agents...")
        # Create a search tool
        self.search_tool = SerperDevTool()
        venue_finder, find_venue_task = self.prepare_venue_finder_agents()
        venue_quality_assurance_agent, quality_assurance_review_task = self.prepare_quality_assurance_agent()
        self.event_planning_crew = Crew(
            agents=[venue_finder, venue_quality_assurance_agent],
            tasks=[find_venue_task, quality_assurance_review_task],
            verbose=True,
            memory=True
            )
        print(f"CrewAI agents initialized.")
                

    def prepare_venue_finder_agents(self):
        print(f"Initializing agent venufinder...") 
        # Define agents
        venue_finder = Agent(
            role="Conference Venue Finder",
            goal="Find the best venue for the upcoming conference",
            backstory=(
                "You are an experienced event planner with a knack for finding the perfect venues. "
                "Your expertise ensures that all conference requirements are met efficiently. "
                "Your goal is to provide the client with the best possible venue options."
            ),
            tools=[self.search_tool],
            verbose=True
        )

        print(f"Initializing Task for Agent: {venue_finder.role}") 
        # Define tasks
        find_venue_task = Task(
            description=(
                "Conduct a thorough search to find the best venue for the upcoming conference. "
                "Consider factors such as capacity, location, amenities, and pricing. "
                "Use online resources and databases to gather comprehensive information."
            ),
            expected_output=(
                "A list of 5 potential venues with detailed information on capacity, location, amenities, pricing, and availability."
            ),
            tools=[self.search_tool],
            agent=venue_finder,
        )

        return venue_finder, find_venue_task


    def prepare_quality_assurance_agent(self):

        print(f"Initializing agent quality assurance...") 
        #  review each venue and ensure it aligns with the client’s expectations.
        venue_quality_assurance_agent = Agent(
            role="Venue Quality Assurance Specialist",
            goal="Ensure the selected venues meet all quality standards and client requirements",
            backstory=(
                "You are meticulous and detail-oriented, ensuring that the venue options provided "
                "are not only suitable but also exceed the client's expectations. "
                "Your job is to review the venue options and provide detailed feedback."
            ),
            tools=[self.search_tool],
            verbose=True
        )

        print(f"Initializing Task for Agent: {venue_quality_assurance_agent.role}") 
        quality_assurance_review_task = Task(
            description=(
                "Review the venue options provided by the Conference Venue Finder. "
                "Ensure that each venue meets all the specified requirements and standards. "
                "Provide a detailed report on the suitability of each venue."
            ),
            expected_output=(
                "A detailed review of the 5 potential venues, highlighting any issues, strengths, and overall suitability."
            ),
            tools=[self.search_tool],
            agent=venue_quality_assurance_agent,
        )

        return venue_quality_assurance_agent, quality_assurance_review_task
    

    def run_event_planning_crew(self):
        print(f"Running CrewAI agents...")
        inputs = {
            "conference_name": "AI Innovations Summit",
            "requirements": "Capacity for 5000, central location, modern amenities, budget up to $50,000"
        }

        result = self.event_planning_crew.kickoff(inputs=inputs)

        print(f"CrewAI agents run completed.")


MyCrewAgent().run_event_planning_crew()
        

