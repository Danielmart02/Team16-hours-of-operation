#create a service client by name using the default session.
import math
import numexpr
import json
import datetime
import sys
import os

import logging

import boto3
import pandas as pd

from langchain_core.tools import tool

from langchain_aws import ChatBedrock
# from langchain_ollama import ChatOllama
from typing import Literal
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from inference import StaffingPredictor
from dining_agent import DiningHallAgent


agent = DiningHallAgent(enable_tracing=True)

# # Example conversation
# questions = [
#     "How many staff members should i hire? its sunny and there are no events today. please give me some ideas of how to manage my employees",
#     "What is my favorite color?"
#
# ]
#
# for question in questions:
#     print(f"\n🤖 Question: {question}")
#     response = agent.ask(question)
#     print(f"📊 Response: {response}")
#     print("=" * 80)
#
# # Clean up
# agent.close()