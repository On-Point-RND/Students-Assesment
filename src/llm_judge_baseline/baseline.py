import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("BASE_URL")


client = OpenAI(api_key=api_key, base_url=base_url)