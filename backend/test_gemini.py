from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents="Give 1 line about tomato leaf disease."
)

print(response.text)