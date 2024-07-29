from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from surprise_travel.tools.custom_tool import MyCustomTool

# Check our tools documentation for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, TXTSearchTool
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

class StrengthTrainingExercise(BaseModel):
    name: str = Field(..., description="Name of exercise")
    description: str = Field(..., description="Description of how to doexercise")
    justification: str = Field(...,description="why this exercise is important for meeting client health and fitness goals")
    equipment: str = Field(..., description="Equipment needed")
    weight: int = Field(..., description="Weight, in lbs.")
    reps: int = Field(..., description="Number of repetitions per set")
    sets: int = Field(..., description="Number of sets")
    additional: str = Field(..., description="Additional instructions")

class StrengthTrainingRoutine(BaseModel):
    workout_location: str = Field(..., description="Where to do the workout")
    time_block: str = Field(..., description="Time block in which to workout")
    duration: int = Field(..., description="Workout duration in minutes")
    body_area_focus: str = Field(..., description="Areas of boy to focus on")
    theme: str = Field(..., description="Strength or endurance")
    exercises: List[StrengthTrainingExercise] = Field(...,description="List of exercises")
    things_to_remember: str = Field(...,description="Things to keep in mind during workout")
class CardioTraining(BaseModel):
    name: str = Field(..., description="Name of exercise")
    description: str = Field(..., description="Detailed description of the exercise")
    location: str = Field(..., description="Location for the exercise")
    time_block: str = Field(...,description="Time block in which to perform the exercise")
    length: int = Field(..., description="Length of exercise")
    target_heart_rate: int = Field(..., description="Target heart rate")

class MobilityTraining(BaseModel):
    name: str = Field(..., description="Name of exercise")
    description: str = Field(..., description="Description of exercise")
    equipment: str = Field(..., description="Equipment needed")
    reps: int = Field(..., description="Number of repetitions per set")
    sets: int = Field(..., description="Number of sets")

class MobilityRoutine(BaseModel):
    preworkout: List[MobilityTraining] = Field(...,description="Pre-workout stretching and mobility")
    postworkout: List[MobilityTraining] = Field(...,description="Post-workout stretching and mobility")


class DietaryGuide(BaseModel):
    guideline: str = Field(..., description="Dietary guideline for the day")
    calories: int = Field(..., description="Caloric target for the day")
    protein: int = Field(..., description="Protein target (g)")
    carbs: int = Field(..., descption="Max carbohydrates (g)")
    fat: int = Field(..., description="Max fat (g)")

class Motivation(BaseModel):
    movement_motivational_message: str = Field(..., description="Motivation to move")
    eating_motivational_message: str = Field(..., description="Motivation for clean eating")

@CrewBase
class WellnessCrew():

    def __init__(self,llm):
        self.llm = llm

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def personal_trainer(self) -> Agent:
        return Agent(
            config=self.agents_config['personal_trainer'],
            tools=[SerperDevTool(),
                   ScrapeWebsiteTool(),
                   TXTSearchTool(txt="./data/journal.txt"),
                   TXTSearchTool(txt="./data/limits.txt")],
            verbose=False,
            allow_delegation=False,
            llm=self.llm
        )


    @agent
    def motivational_coach(self) -> Agent:
        return Agent(
            config=self.agents_config['motivational_coach'],
            tools=[SerperDevTool(),
                   ScrapeWebsiteTool(),
                   TXTSearchTool(txt="./data/journal.txt")],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )


    @agent
    def physical_therapist(self) -> Agent:
        return Agent(
            config=self.agents_config['physical_therapist'],
            tools=[SerperDevTool(),
                   ScrapeWebsiteTool(),
                   TXTSearchTool(txt="./data/journal.txt")],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    @agent
    def nutritionist(self) -> Agent:
         return Agent(
             config=self.agents_config['nutritionist'],
             tools=[SerperDevTool(),
                    ScrapeWebsiteTool(),
                    TXTSearchTool(txt="./data/journal.txt")],
             verbose=True,
             memory=True,
             allow_delegation=False
         )

    @task
    def strength_focus_area_task(self) -> Task:
        return Task(
            config=self.tasks_config['strength_focus_area_task'],
            agent=self.personal_trainer(),
            memory=True
        )

    @task
    def strength_exercise_research_task(self) -> Task:
        return Task(
            config=self.tasks_config['strength_exercise_research_task'],
            tools=[SerperDevTool(),ScrapeWebsiteTool()],
            agent=self.personal_trainer(),
            memory=True
        )

    @task
    def strength_routine_task(self) -> Task:
        return Task(
            config=self.tasks_config['strength_routine_task'],
            tools=[],
            agent=self.personal_trainer(),
            memory=True,
            output_json=StrengthTrainingRoutine
        )

    @task
    def cardio_research_task(self) -> Task:
         return Task(
             config=self.tasks_config['cardio_research_task'],
             agent=self.personal_trainer(),
             memory=True
         )

    @task
    def cardio_routine_task(self) -> Task:
         return Task(
             config=self.tasks_config['cardio_routine_task'],
             agent=self.personal_trainer(),
             output_json=CardioTraining,
             memory=True
         )

    @task
    def mobility_workout_research_task(self) -> Task:
         return Task(
             config=self.tasks_config['mobility_workout_research_task'],
             agent=self.physical_therapist(),
             memory=True
         )

    @task
    def mobility_routine_task(self) -> Task:
         return Task(
             config=self.tasks_config['mobility_routine_task'],
             agent=self.physical_therapist(),
             memory=True,
             output_json=MobilityRoutine
         )

    @task
    def motivation_task(self) -> Task:
        return Task(
            config=self.tasks_config['motivation_task'],
            agent=self.motivational_coach(),
            memory=True,
            output_json=Motivation
        )

    @task
    def nutrition_plan_task(self) -> Task:
         return Task(
             config=self.tasks_config['nutrition_plan_task'],
             agent=self.nutritionist(),
             output_json=DietaryGuide
         )

    @crew
    def strength_crew(self) -> Crew:
        """Creates the Strength training crew"""

        return Crew(
            agents=[self.personal_trainer()],
            tasks=[self.strength_focus_area_task(),
                  self.strength_exercise_research_task(),
                  self.strength_routine_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )


    @crew
    def cardio_crew(self) -> Crew:
        """Creates the cardio crew"""
        return Crew(
            agents=[self.personal_trainer()],
            tasks=[self.cardio_research_task(),
                   self.cardio_routine_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )


    @crew
    def pt_crew(self) -> Crew:
        """Creates the cardio crew"""
        return Crew(
            agents=[self.physical_therapist()],
            tasks=[self.mobility_workout_research_task(),
                   self.mobility_routine_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )

    @crew
    def nutritionist_crew(self) -> Crew:
        """Creates the nutritionist crew"""
        return Crew(
            agents=[self.nutritionist()],
            tasks=[self.nutrition_plan_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )

    @crew
    def motivation_crew(self) -> Crew:
        """Creates the cardio crew"""
        return Crew(
            agents=[self.motivational_coach()],
            tasks=[self.motivation_task()],
            process=Process.sequential,
            planning=True,
            memory=True,
            verbose=2
        )

