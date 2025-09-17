import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
import argparse
from tqdm import tqdm

# get envs for openai client
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# rate limit
MAX_CONCURENT_REQUESTS = 10
semaphore = asyncio.Semaphore(MAX_CONCURENT_REQUESTS)


def get_data(data_type: str) -> list[str]:
    """
    load cv, cover letter or presentation for llm scoring
    """
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent.parent / "data"

    df = pd.read_parquet(data_dir + "/dataset.parquet")
    data = df[data_type].tolist()

    return data


async def ask(prompt: str) -> int:
    """
    get response from llm with marker for data
    """
    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Тебе дано резюме студента, который проходит отобр на летнюю школу по искусственому интеллекту. Твоя задача оценить его по десятибалльной шкале"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
        )
        return response.choices[0].message.content


async def main():
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="cv", choices=["cv", "letter", "presentation"])
    parser.add_argument("--output_file_name", type=str)
    args = parser.parse_args()

    # get data
    current_dir = Path(__file__).resolve().parent
    data_dir = current_dir.parent.parent / "data"
    df = pd.read_parquet(data_dir / "dataset.parquet").iloc[:100]
    data = df[args.data_type].tolist()

    # get answers from llm
    tasks = [ask(p) for p in data]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="LLM requests"):
        res = await coro
        results.append(res)

    # save markers
    df["llm_markers"] = results
    df = df[[args.data_type, "llm_markers"]]
    output_file = data_dir / f"{args.output_file_name}.csv"
    df.to_csv(output_file, index= False)
    print(f"✅ Файл сохранен")


if __name__ == "__main__":
    asyncio.run(main())