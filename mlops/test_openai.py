import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("No encuento OPENAI_API_KEY. Revisa tu .env")

client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Eres un asistente de IA."},
        {"role": "user",   "content": "Hola"}
    ],
)

print(response.choices[0].message.content)
