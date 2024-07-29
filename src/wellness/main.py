#!/usr/bin/env python
import sys

from crewai.crews import CrewOutput
from langchain_community.llms import ollama
from langchain_community.llms import openai
from wellness.crew import WellnessCrew
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import dotenv
import os
from json import dumps

openai_llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                        model=os.getenv("OPENAI_MODEL_NAME"))

groq_llm = ChatGroq(
  api_key=os.getenv('GROQ_API_KEY'),
#  model="llama-3.1-70b-versatile",
  model="llama-3.1-8b-instant",
  max_tokens=2000,
  max_retries=1
)


ollama_llm = ollama.Ollama(
    model = "llama3.1",
    base_url = "http://localhost:11434")

inputs = {
 'time_blocks_with_location': 'home (8 Via San Inigo, Orinda, CA) at 6am-8am and gym at 11:30am-2:30apm',

  'motivation_level': 'medium',
  'workouts_yesterday': 'cardio, back and biceps',
  'eating_yesterday': 'two high fiber chicken quesadillas for lunch, three small high fiber cheese quesadillas for a late dinner',
  'aches_and_pains': 'none',
  'morning_weight': '175lbs',
  'target_weight': '150lbs',

  'physical_issues':'old rotor cuff injuries in both shoulders',
  'general_goals': 'Hit a target body weight of 150 and a very healthy BMI. Build lean muscle mass. Increase weight and max PRs.  Try to eat under 1500 calories a day'
}

def run_strength():

  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = (WellnessCrew(llm=openai_llm).
                        strength_crew().
                        kickoff(inputs=inputs))
  return result

def run_cardio():
  # Replace with your inputs, it will automatically interpolate any tasks and agents information
  result: CrewOutput = WellnessCrew(llm=openai_llm).cardio_crew().kickoff(inputs=inputs)
  return result


workout = """
{
  "workout_time_and_location": "Gym",
  "workout_duration": "180 minutes",
  "workout_time_block": "11:30am-2:30pm",
  "areas_of_the_body_to_focus_on": "Upper Body, Core, Lower Body",
  "theme": "Strength",
  "exercises": [
    {
      "exercise_name": "Dumbbell Reverse Flyes",
      "weight": "5-15 lbs",
      "reps": 12,
      "sets": 3,
      "equipment_needed": "Dumbbells",
      "muscle_groups_activated": "Upper Back, Rear Deltoids",
      "description": "Stand with feet shoulder-width apart, bend slightly at the hips, hold dumbbells with palms facing each other, and raise your arms out to the side.",
      "video_link": "https://www.youtube.com/watch?v=V5vD6gP8D0U"
    },
    {
      "exercise_name": "Seated Rows",
      "weight": "20-50 lbs",
      "reps": 10,
      "sets": 3,
      "equipment_needed": "Cable Machine or Resistance Bands",
      "muscle_groups_activated": "Upper Back, Lats",
      "description": "Sit at the cable machine, grasp the handle, and pull it towards your torso, squeezing shoulder blades together.",
      "video_link": "https://www.youtube.com/watch?v=GQOZ4WgNg6E"
    },
    {
      "exercise_name": "Single-Arm Dumbbell Chest Press",
      "weight": "10-25 lbs",
      "reps": 10,
      "sets": 3,
      "equipment_needed": "Dumbbells, Bench",
      "muscle_groups_activated": "Chest, Triceps",
      "description": "Lie back on a bench, hold a dumbbell in one hand, and press it upward until your arm is extended.",
      "video_link": "https://www.youtube.com/watch?v=G5TXsD8Nf8g"
    },
    {
      "exercise_name": "Hammer Curl",
      "weight": "10-20 lbs",
      "reps": 12,
      "sets": 3,
      "equipment_needed": "Dumbbells",
      "muscle_groups_activated": "Biceps",
      "description": "Stand with dumbbells at your sides, palms facing in, and curl the weights up towards your shoulders.",
      "video_link": "https://www.youtube.com/watch?v=Zz8x8f5m3D4"
    },
    {
      "exercise_name": "Bent-Over Row",
      "weight": "10-25 lbs",
      "reps": 10,
      "sets": 3,
      "equipment_needed": "Dumbbells",
      "muscle_groups_activated": "Upper Back, Lats, Biceps",
      "description": "Bend at the hips with dumbbells in hand, pull the weights towards your waist while keeping your back straight.",
      "video_link": "https://www.youtube.com/watch?v=3G8X0x3fI5M"
    },
    {
      "exercise_name": "Incline Push-up",
      "weight": "Bodyweight",
      "reps": 10,
      "sets": 3,
      "equipment_needed": "Bench or Stable Elevated Surface",
      "muscle_groups_activated": "Chest, Shoulders, Triceps",
      "description": "Place your hands on an elevated surface and lower your body until your chest nearly touches it.",
      "video_link": "https://www.youtube.com/watch?v=nDiC4U4GVr8"
    },
    {
      "exercise_name": "Plank Shoulder Tap",
      "weight": "Bodyweight",
      "reps": 10,
      "sets": 3,
      "equipment_needed": "None",
      "muscle_groups_activated": "Core, Shoulders",
      "description": "Hold a plank position and tap each shoulder with the opposite hand, maintaining balance.",
      "video_link": "https://www.youtube.com/watch?v=5p7g8z8pR0I"
    },
    {
      "exercise_name": "Band Pull Aparts",
      "weight": "Resistance Band",
      "reps": 12,
      "sets": 3,
      "equipment_needed": "Resistance Band",
      "muscle_groups_activated": "Upper Back, Rear Deltoids",
      "description": "Hold a resistance band at shoulder height with both hands and pull it apart, squeezing shoulder blades together.",
      "video_link": "https://www.youtube.com/watch?v=5p7g8z8pR0I"
    },
    {
      "exercise_name": "Single-Leg Deadlift",
      "weight": "5-15 lbs",
      "reps": 10,
      "sets": 3,
      "equipment_needed": "Dumbbells",
      "muscle_groups_activated": "Hamstrings, Glutes, Lower Back",
      "description": "Stand on one leg and hinge forward at the hips while lowering a dumbbell towards the ground.",
      "video_link": "https://www.youtube.com/watch?v=3bHR1Nc1O4A"
    },
    {
      "exercise_name": "Farmer’s Carry",
      "weight": "20-40 lbs",
      "reps": "N/A",
      "sets": 3,
      "equipment_needed": "Dumbbells or Kettlebells",
      "muscle_groups_activated": "Core, Grip, Shoulders",
      "description": "Hold a weight in each hand and walk a set distance, maintaining good posture.",
      "video_link": "https://www.youtube.com/watch?v=8S92wCq0sR8"
    }
  ],
  "thoughts": "Focus on maintaining proper form throughout each exercise, especially with shoulder stability in mind. Emphasize controlled movements to avoid straining the rotator cuff. Encourage gradual progression in weights and reps as strength improves."
}
"""
inputs_with_workout = inputs
inputs_with_workout['workout'] = workout

def run_pt():
  result: CrewOutput = WellnessCrew(llm=openai_llm).pt_crew().kickoff(inputs=inputs_with_workout)
  return result

pt = """
{
  "pre_workout_routine": [
    {
      "name": "Standing Arm Swings",
      "sets": "1",
      "reps": "30-60 seconds",
      "equipment_needed": "None",
      "description": "Stand tall with your arms by your sides. Engage your core and swing your arms forward until they’re as high as you can go, then return to the starting position.",
      "body_part_supported": "Shoulders and upper body"
    },
    {
      "name": "Shoulder Pass-Through",
      "sets": "1",
      "reps": "5",
      "equipment_needed": "Broomstick or PVC pipe",
      "description": "Stand with feet shoulder-width apart, hold the stick with an overhand grip, and raise it above your head, keeping your arms straight.",
      "body_part_supported": "Shoulders and upper back"
    },
    {
      "name": "High-to-Low Rows",
      "sets": "2-3",
      "reps": "10",
      "equipment_needed": "Resistance band or cable machine",
      "description": "Kneel and pull the band towards your body while keeping your torso straight, squeezing your shoulder blades together.",
      "body_part_supported": "Upper back and shoulders"
    },
    {
      "name": "Reverse Fly",
      "sets": "3",
      "reps": "10",
      "equipment_needed": "Light dumbbells",
      "description": "Bend forward at the waist with a dumbbell in each hand and raise your arms away from your body, squeezing shoulder blades together.",
      "body_part_supported": "Shoulders and upper back"
    },
    {
      "name": "Rotation with Dumbbell",
      "sets": "2-3",
      "reps": "12",
      "equipment_needed": "Light dumbbell",
      "description": "Stand with a dumbbell in one hand at shoulder height, rotate your shoulder to bring the weight up towards the ceiling.",
      "body_part_supported": "Shoulders and core"
    }
  ],
  "post_workout_routine": [
    {
      "name": "Cross-Body Shoulder Stretch",
      "sets": "1",
      "reps": "30 seconds each side",
      "equipment_needed": "None",
      "description": "Reach one arm across your body and gently pull it with the opposite hand to stretch the shoulder.",
      "body_part_supported": "Shoulders"
    },
    {
      "name": "Sleeper Stretch",
      "sets": "3",
      "reps": "30 seconds each side",
      "equipment_needed": "None",
      "description": "Lie on the affected side, bend the elbow to 90 degrees, and gently guide the arm towards the floor.",
      "body_part_supported": "Shoulders"
    },
    {
      "name": "Doorway Stretch",
      "sets": "2-3",
      "reps": "30 seconds each side",
      "equipment_needed": "Doorway",
      "description": "Stand in a doorway with arms raised at 90 degrees and gently lean into the stretch.",
      "body_part_supported": "Chest and shoulders"
    },
    {
      "name": "Child's Pose",
      "sets": "1",
      "reps": "3-5 breaths",
      "equipment_needed": "Exercise mat",
      "description": "Kneel, extend your arms forward, and lower your torso to stretch the shoulders and lats.",
      "body_part_supported": "Shoulders and back"
    },
    {
      "name": "Chest Expansion",
      "sets": "3-5",
      "reps": "30 seconds",
      "equipment_needed": "Towel or exercise band",
      "description": "Hold a towel behind your back and pull it to open up the chest and stretch the shoulders.",
      "body_part_supported": "Chest and shoulders"
    }
  ]
}"""

def run_motivation():
  result: CrewOutput = WellnessCrew(llm=openai_llm).motivational_crew().kickoff(inputs=inputs_with_workout)
  print(result.raw)


def run_day():
  team = WellnessCrew(llm=openai_llm)

  strength: CrewOutput = (WellnessCrew(llm=openai_llm).
                        strength_crew().
                        kickoff(inputs=inputs))
  cardio: CrewOutput = (WellnessCrew(llm=openai_llm).
                        cardio_crew().
                        kickoff(inputs=inputs))

  nutrition: CrewOutput = (WellnessCrew(llm=openai_llm).
                           nutritionist_crew().
                           kickoff(inputs=inputs))

  inputs['workout'] = strength.raw

  pt: CrewOutput = (WellnessCrew(llm=openai_llm).
                        pt_crew().
                        kickoff(inputs=inputs))

  inputs['diet'] = nutrition.raw

  motivation: CrewOutput = (WellnessCrew(llm=openai_llm).
                        motivation_crew().
                        kickoff(inputs=inputs))



  print(motivation.raw)
  print(nutrition.raw)
  print(cardio.raw)
  print(strength.raw)
  print(pt.raw)