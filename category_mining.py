import os
import json
import re
import random
from itertools import combinations
from typing import Dict, Any, List

# ----------------------------
# 1. LLM Inferer
# ----------------------------
from agents import OpenRouterInferer


# ----------------------------
# 2. Prompt Templates
# ----------------------------

# Stage 1: Discovery → produce name + evaluation QUESTION
DISCOVERY_SYSTEM = (
    "You are an expert evaluator of {doc_type}. "
    "Given two {doc_type_plural}, identify up to three MOST discriminative features that clearly distinguish them. "
    "For each feature, provide:\n"
    "- A short NAME (2-4 words)\n"
    "- A precise EVALUATION QUESTION that could be used to assess this feature in future candidates "
    "(e.g., 'Can the candidate...?', 'How well does the candidate...?').\n"
    "Focus on objective, observable traits. Avoid vague terms. "
    "Output STRICTLY as a JSON list: [{{\"name\": \"...\", \"question\": \"...\"}}, ...]. "
    "NO extra text, NO markdown, ONLY JSON."
)

# Stage 2: Merging → compare QUESTIONS (not descriptions)
MERGING_SYSTEM = (
    "You are a taxonomy curator. You will be given:\n"
    "- A CANDIDATE category (name + evaluation question)\n"
    "- A list of EXISTING categories (name + evaluation question)\n\n"
    "Your task: Decide if the candidate is SEMANTICALLY EQUIVALENT to any existing category.\n"
    "Rules:\n"
    "- If YES: output JSON {{\"match\": true, \"existing_name\": \"ExactName\"}}\n"
    "- If NO: output JSON {{\"match\": false}}\n"
    "Be strict: only match if they assess the same underlying trait using equivalent questions. "
    "Output ONLY valid JSON. No explanations."
)

MERGING_USER_TEMPLATE = (
    "CANDIDATE CATEGORY:\n"
    "Name: {candidate_name}\n"
    "Question: {candidate_question}\n\n"
    "EXISTING CATEGORIES:\n{existing_list_str}\n\n"
    "Now decide if the candidate matches any existing category."
)


# ----------------------------
# 3. Robust JSON Parser
# ----------------------------
def extract_and_parse_json(llm_output: str):
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", llm_output)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = llm_output.strip()

    if not (candidate.startswith("[") or candidate.startswith("{")):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1:
            start = candidate.find("[")
            end = candidate.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = candidate[start:end+1]
        else:
            raise ValueError("No JSON structure found")

    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON decode error: {e}")


# ----------------------------
# 4. Core Mining Function (Two-Stage with Questions)
# ----------------------------
def mine_categories_from_pairs(
    input_dir: str,
    output_file: str,
    doc_type: str,
    api_key: str,
    num_pairs: int = 100,
    max_retries: int = 3,
    model: str = "qwen/qwen3-next-80b-a3b-instruct:free"
):
    doc_type_plural_map = {
        "resume": "resumes",
        "motivation_letter": "motivation letters",
        "presentation": "presentations"
    }
    doc_type_plural = doc_type_plural_map.get(doc_type, doc_type + "s")

    # Load documents
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if len(files) < 2:
        raise ValueError(f"Need at least 2 documents in {input_dir}")
    
    documents = {}
    for f in files:
        try:
            with open(os.path.join(input_dir, f), 'r', encoding='utf-8') as file:
                documents[f] = file.read()
        except Exception as e:
            print(f"Warning: Skipping {f} due to read error: {e}")

    if len(documents) < 2:
        raise ValueError("Not enough readable documents.")

    all_pairs = list(combinations(documents.items(), 2))
    sampled_pairs = random.sample(all_pairs, min(num_pairs, len(all_pairs)))

    # Registry: {name: {"question": str, "count": int}}
    category_registry: Dict[str, Dict[str, Any]] = {}

    inferer = OpenRouterInferer(api_key=api_key, model=model)
    discovery_system = DISCOVERY_SYSTEM.format(
        doc_type=doc_type,
        doc_type_plural=doc_type_plural
    )
    merging_system = MERGING_SYSTEM

    for i, ((name_a, doc_a), (name_b, doc_b)) in enumerate(sampled_pairs):
        print(f"Processing pair {i+1}/{len(sampled_pairs)}: {name_a} vs {name_b}")

        # ===== STAGE 1: Discovery (no history) =====
        discovery_user = (
            f"Document A:\n{doc_a[:3500]}\n\n"
            f"Document B:\n{doc_b[:3500]}"
        )

        candidates = []
        for attempt in range(max_retries):
            try:
                raw_resp = inferer.forward(discovery_system, discovery_user)
                parsed = extract_and_parse_json(raw_resp)
                if isinstance(parsed, dict):
                    parsed = [parsed]
                if isinstance(parsed, list):
                    candidates = [
                        {
                            "name": str(item["name"]).strip(),
                            "question": str(item.get("question", "")).strip()
                        }
                        for item in parsed
                        if isinstance(item, dict) and "name" in item and "question" in item
                    ]
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  ❌ Discovery failed: {e}")
                continue

        if not candidates:
            continue

        # ===== STAGE 2: Merging =====
        existing_list = [
            {"name": name, "question": data["question"]}
            for name, data in category_registry.items()
        ]

        for cand in candidates:
            cand_name = cand["name"]
            cand_question = cand["question"] or "(no question)"

            if not cand_name or not cand_question:
                continue

            if not existing_list:
                category_registry[cand_name] = {"question": cand_question, "count": 1}
                existing_list.append({"name": cand_name, "question": cand_question})
                continue

            # Build existing list string
            existing_str = "\n".join(
                f"- Name: {item['name']}\n  Question: {item['question']}"
                for item in existing_list
            )

            merging_user = MERGING_USER_TEMPLATE.format(
                candidate_name=cand_name,
                candidate_question=cand_question,
                existing_list_str=existing_str
            )

            merged = False
            for attempt in range(max_retries):
                try:
                    raw_merge = inferer.forward(merging_system, merging_user)
                    merge_result = extract_and_parse_json(raw_merge)

                    if isinstance(merge_result, dict):
                        if merge_result.get("match") is True:
                            existing_name = str(merge_result.get("existing_name", "")).strip()
                            if existing_name in category_registry:
                                category_registry[existing_name]["count"] += 1
                                merged = True
                                break
                        elif merge_result.get("match") is False:
                            category_registry[cand_name] = {"question": cand_question, "count": 1}
                            existing_list.append({"name": cand_name, "question": cand_question})
                            merged = True
                            break
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"    ⚠️ Merging failed for '{cand_name}': {e}")
                    pass

            if not merged:
                # Fallback: treat as new
                category_registry[cand_name] = {"question": cand_question, "count": 1}
                existing_list.append({"name": cand_name, "question": cand_question})

    # Save result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(category_registry, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Final registry: {len(category_registry)} categories. Saved to {output_file}")


# ----------------------------
# 5. CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-stage category mining using evaluation QUESTIONS.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--doc_type",
        required=True,
        choices=["resume", "motivation_letter", "presentation"]
    )
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--num_pairs", type=int, default=100)
    parser.add_argument("--model", type=str, default=100)

    args = parser.parse_args()

    mine_categories_from_pairs(
        input_dir=args.input_dir,
        output_file=args.output_file,
        doc_type=args.doc_type,
        api_key=args.api_key,
        num_pairs=args.num_pairs,
        model = args.model
    )