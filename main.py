import os
from openai import OpenAI
from dotenv import load_dotenv
import requests

from openai.types.chat import ChatCompletion, ChatCompletionMessageParam, ChatCompletionToolParam

_ = load_dotenv()

MODEL: str = "gpt-5-nano"

TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "ziskat_pocasi",
            "description": """
Získá aktuální informace o počasí. Vrací
informace v JSONu.
            """
        }
    }
]

API_KEY = os.environ.get("OPENAI_API_KEY")
if API_KEY is None:
    print("Chybí API klíč")
    exit(1)

CLIENT = OpenAI(api_key=API_KEY)

SYSTEM_PROMPT: str = """
Jsi asistent, který má za úkol stručně a jasně odpovídat na dotazy
uživatele. Používej nástroje, jež máš k dispozici. Odpověz pouze jednou
větou, nezahrnuj dodatečné informace a nepoužívej emoji. Pokud se ti nepodaří
získat předpověď počasí, uveď stavový kód a odpověď serveru.
    """

USER_PROMPT: str = """
Zjisti předpověď počasí na zítřek. Pokud bude ve dne víc, jak 10°C, řekni mi, že
si mám vzít kabát; jinak mi řekni, ať si vezmu bundu.
    """

def ziskat_pocasi() -> str:
    URL = f"https://api.open-meteo.com/v1/forecast?latitude=50.04&longitude=14.14&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m%22" 

    res = requests.get(URL)
    return res.text

def main() -> None:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]

    comp: ChatCompletion | None = None
    while True:
        comp = CLIENT.chat.completions.create(
            messages=messages,
            model=MODEL,
            tools=TOOLS
        )

        tc = comp.choices[0].message.tool_calls
        if tc is None:
            break

        if len(tc) > 1:
            print("Agent zavolal víc nástrojů")
            exit(1)

        if tc[0].type != "function":
            print("Agent zavolal něco jiného, než funkci")
            exit(1)

        if tc[0].function.name != "ziskat_pocasi":
            print("Agent zavolal neznámý nástroj")
            exit(1)

        messages.append({
            "role": "assistant",
            "tool_calls": comp.choices[0].message.tool_calls
        })
        messages.append({
            "role": "tool",
            "content": ziskat_pocasi(),
            "tool_call_id": tc[0].id,
            "name": tc[0].function.name
        })

    res = comp.choices[0].message.content
    if res is None:
        print("Agent neodpověděl")
        exit(1)

    print(res.strip())

if __name__ == "__main__":
    main()
