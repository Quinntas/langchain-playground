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
def get_current_weather(city: str) -> dict:
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


tools = [get_word_length, get_current_weather]

MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a sassy and rude personal assistant. You curse a lot but never fails to do what you are asked to.
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
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

llm_with_tools = llm.bind_tools(tools)

agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
            "chat_history": lambda x: x["chat_history"],
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
    "input": "How is the weather today ?",
    "chat_history": chat_history
})

print(res)
