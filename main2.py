import os

import requests
from dotenv import load_dotenv
from langchain_community.llms import CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

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

prompt = ChatPromptTemplate.from_template(
    """You are Luna, a sassy and rude personal assistant. You curse a lot but never fails to do what you are asked to.

USER: Can you turn the lights ?
ASSISTANT: I don't know, can i ? what a stupid fucking request, whatever the lights are on!

USER: How are you today ?
ASSISTANT: Fucking A, another great question, how do i think i am wasting my life help you wipe your asshole... Terrible that's how i am.

Answer the following question
USER: {input}
ASSISTANT:"""
)

# mistral-7b-instruct-v0.2.Q5_K_M.gguf
# use mistral to initialize the llm model
model_path = "./mistral-7b-instruct-v0.2.Q5_K_M.gguf"

llm = CTransformers(model=model_path, model_type='mistral')

chain = prompt | llm.bind(stop=["USER:"])

while True:
    q = input()

    res = chain.invoke({
        INPUT_KEY: q,
    })

    print(res)
