import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
import argparse
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import json
from tenacity import retry, stop_after_attempt, wait_fixed


# get envs for openai client
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = os.getenv("MODEL")
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# system_prompt
SYSTEM_PROMPT = """
# РОЛЬ
Ты эксперт по оценке кандидатур студентов на летнюю школу SMILES по искусственному интеллекту. 

# ЗАДАЧА
Твоя задача — помочь определить, насколько кандидат подходит для участия в программе. 
Тебе будет дан один из следующих документов: резюме (CV), мотивационное письмо (Cover Letter) или презентация. 
Исходя из предоставленного документа:
1. Проанализируй сильные и слабые стороны кандидата, опыт, навыки и потенциал.
2. Убедись что кандидат подходит именно под данную тематику школы: школа по искусственному интелекту с упором в исследования.
3. Поставь финальную оценку по десятибалльной шкале.
Не придумывай информацию, оценивай только на основе предоставленного текста.

# ФОРМАТ ОТВЕТА
Ответь в формате JSON:
{
    "analysis": "анализ кандидатуры",
    "score": "финальная оценка по десятибалльной шкале, только целые числа"
}
"""

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

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
async def ask(prompt: str) -> int:
    """
    get response from llm with marker for data
    if prompt NA return score -1 and analysis -
    """

    if type(prompt) != str:
        return '{"analysis": "-",  "score": -1}'

    async with semaphore:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            response_format={
                "type": "json_object"
            }
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
    df = pd.read_parquet(data_dir / "dataset.parquet")
    df = df.reset_index()
    data = df[args.data_type].tolist()

    # get answers from llm
    tasks = [ask(p) for p in data]
    results = await tqdm_asyncio.gather(*tasks)

    # get llm analysis(CoT) and score
    df["llm_analysis"] = [json.loads(r)["analysis"] for r in results]
    df["llm_score"] = [int(json.loads(r)["score"]) for r in results]
    df = df[["name", args.data_type, "llm_analysis", "llm_score"]]

    # save results
    output_file = data_dir / f"{args.output_file_name}.csv"
    df.to_csv(output_file, index= False)
    print(f"✅ Файл сохранен\nКоличество пустых кейсов {(df.llm_score == -1).sum()}")


if __name__ == "__main__":
    asyncio.run(main())