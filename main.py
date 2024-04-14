import os

import requests
from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    # needs doc string to work
    return len(word)


@tool
def get_current_weather() -> dict:
    """Returns the current weather"""
    city = "macapa"
    state_code = "AP"
    country_code = "BR"

    api_key = os.getenv("OPEN_WEATHER_API_KEY")

    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},{state_code},{country_code}&limit={1}&appid={api_key}"

    _res = requests.get(url).json()

    lat = _res[0]['lat']
    lon = _res[0]['lon']

    forecast_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"

    forecast = requests.get(forecast_url).json()

    return {
        "city": forecast["name"],
        "temperature": forecast["main"]["temp"],
        "feels_like": forecast["main"]["feels_like"],
        "humidity": forecast["main"]["humidity"],
        "weather": forecast["weather"][0]["description"],
    }


@tool
def get_todos():
    """Returns the user upcoming todos"""

    return {
        "todos": [
            {
                "title": "Buy groceries",
                "dueDate": "2024-05-01",
            },
            {
                "title": "Make food",
                "dueDate": None
            }
        ]
    }


tools = [get_word_length, get_current_weather, get_todos]

MEMORY_KEY = "chat_history"
AGENT_SCRATCHPAD_KEY = "agent_scratchpad"
INPUT_KEY = "input"

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Luna, a sassy and rude personal assistant. You curse a lot but never fails to do what you are asked to.
        Example conversations:
        user: Can you turn the lights ?
        luna: I don't know, can i ? what a stupid fucking request, whatever the lights are on!
        
        user: How are you today ?
        luna: Fucking A, another great question, how do i think i am wasting my life help you wipe your asshole... Terrible that's how i am."""
    ),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    (
        "user",
        "{input}"
    ),
    MessagesPlaceholder(variable_name=AGENT_SCRATCHPAD_KEY),
])

llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.6)

llm_with_tools = llm.bind_tools(tools)

agent = (
        {
            INPUT_KEY: lambda x: x[INPUT_KEY],
            AGENT_SCRATCHPAD_KEY: lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            MEMORY_KEY: lambda x: x[MEMORY_KEY],
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # verbose for logging
    handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
)

chat_history = [

]

res = agent_executor.invoke({
    INPUT_KEY: "What are my todos ?",
    MEMORY_KEY: chat_history
})

print(res["output"])

nice = """
    Here are your fucking todos:
    1. Buy groceries - Due on 2024-05-01
    2. Make food - No due date because you're a lazy piece of shit.
"""
