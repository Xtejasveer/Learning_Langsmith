from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from ddgs import DDGS


os.environ['LANGCHAIN_PROJECT'] = "ReAct Agent"
load_dotenv()

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return str(results) if results else "No results found."


@tool
def get_weather_data(city: str) -> str:
     """Fetches the current weather data for a given city"""
     api_key = os.environ["OPENWEATHER_API_KEY"]
     url = f'https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
     response = requests.get(url)
     return str(response.json())

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=2,
    temperature= 0
)


# Step 2: Pull the ReAct prompt from LangChain Hub
template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
# pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5
)

# What is the release date of Dhadak 2?
# What is the current temp of gurgaon
# Identify the birthplace city of Kalpana Chawla (search) and give its current temperature.

# Step 5: Invoke
response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})
print(response)

print(response['output'])