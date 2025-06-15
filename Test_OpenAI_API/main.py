from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from dotenv import load_dotenv
import os


# Load environment variables from .env
load_dotenv()

# Securely retrieve the API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "response": ""})

@app.post("/", response_class=HTMLResponse)
async def chat(request: Request, user_input: str = Form(...)):
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[{"role": "user", "content": user_input}],
            max_tokens=150,
        )
        response = completion.choices[0].message.content.strip()
    except Exception as e:
        response = f"Error: {str(e)}"

    return templates.TemplateResponse("index.html", {"request": request, "response": response, "user_input": user_input})

