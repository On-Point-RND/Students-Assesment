#!/usr/bin/env python3
"""
Batched Document Evaluation Script (Single-threaded with Resume Support)

This script evaluates CVs, motivation letters, and presentations using AI-based scoring
with batching to reduce API calls. It groups multiple categories per API call.

It supports:
- Appending results after each document
- Resuming from interruption (skips already processed docs)

Usage:
    python evaluate_documents_batched.py --categories <json_path> --folder <folder_path> 
           --output <output_csv> --n <num_categories> --api_key <api_key> --model <model_name>
           [--batch_size <categories_per_call>]

Example:
    python evaluate_documents_batched.py --categories cat_mining_results/cv_categories.json \
           --folder data/raw/cvs_an --output cv_scores.csv --n 100 \
           --api_key "your-api-key" --model "openai/gpt-3.5-turbo" --batch_size 20
"""

import argparse
import json
import os
import glob
import csv
import time
from typing import Dict, List, Tuple, Any, Optional, Set
import sys
from pathlib import Path
import tiktoken  # For token counting
from datetime import timedelta

# Import the OpenRouterInferer class
try:
    from agents import OpenRouterInferer
except ImportError:
    print("Error: Could not import OpenRouterInferer from agents.py")
    print("Make sure agents.py is in the same directory or in your PYTHONPATH")
    sys.exit(1)


class DocumentEvaluator:
    """Evaluates documents against selected categories using AI with batching and resume support."""
    
    def __init__(self, api_key: str, model: str, batch_size: int = 20):
        self.inferer = OpenRouterInferer(api_key=api_key, model=model)
        self.batch_size = batch_size
        self.token_encoder = self._get_token_encoder(model)
        self.stats = {
            'total_calls': 0,
            'total_documents': 0,
            'total_categories': 0,
            'start_time': None,
            'errors': 0,
            'cache_hits': 0,
            'completed_docs': 0  # Track completed during this run
        }
        self.cache = {}
        
    def _get_token_encoder(self, model: str):
        try:
            if 'gpt' in model.lower():
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            elif 'claude' in model.lower():
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as e:
            print(f"Warning: tiktoken error ({e}). Using character count for token estimation.")
            return None
    
    def estimate_tokens(self, text: str) -> int:
        if self.token_encoder:
            return len(self.token_encoder.encode(text))
        else:
            return len(text) // 4
    
    def load_categories(self, json_path: str) -> Dict[str, Dict[str, Any]]:
        with open(json_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        return categories
    
    def select_top_categories(self, categories: Dict[str, Dict[str, Any]], n: int) -> List[Tuple[str, Dict[str, Any]]]:
        sorted_categories = sorted(
            categories.items(),
            key=lambda x: x[1].get('count', 0),
            reverse=True
        )
        return sorted_categories[:n]
    
    def read_documents(self, folder_path: str, max_docs: Optional[int] = None) -> Dict[str, str]:
        documents = {}
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        
        if max_docs:
            txt_files = txt_files[:max_docs]
        
        for file_path in txt_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                filename = os.path.basename(file_path)
                documents[filename] = content
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Loaded {len(documents)} documents from {folder_path}")
        return documents

    def get_completed_documents(self, output_path: str) -> Set[str]:
        """Read already processed document names from output CSV."""
        if not os.path.exists(output_path):
            return set()
        
        completed = set()
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return set()
                doc_index = 0  # document_name is first column
                for row in reader:
                    if row and len(row) > doc_index:
                        completed.add(row[doc_index])
        except Exception as e:
            print(f"Warning: Could not read existing output file {output_path}: {e}")
            return set()
        
        print(f"Found {len(completed)} already processed documents. Resuming...")
        return completed

    def create_batched_prompt(self, document_content: str, 
                             category_batch: List[Tuple[str, Dict[str, Any]]]) -> str:
        categories_section = "CATEGORIES TO EVALUATE:\n"
        #MAX_DOC_LENGTH = 2000  # characters (~500 tokens)
        #document_content = document_content[:MAX_DOC_LENGTH]
        for i, (cat_name, cat_info) in enumerate(category_batch, 1):
            question = cat_info.get('question', '')
            categories_section += f"{i}. {cat_name}: {question}\n"
        
        prompt = f"""Evaluate the following document against these categories:

{categories_section}

DOCUMENT CONTENT:
{document_content}

For each category, evaluate how strongly it is present in the document on a scale from 0 to 5, where:
- 0: Not present at all
- 1: Very weak presence
- 2: Weak presence  
- 3: Moderate presence
- 4: Strong presence
- 5: Very strong presence

Provide your response as a JSON object with category names as keys and scores (0-5 integers) as values.
Example format:
{{
  "Publication Record": 4,
  "Teaching Experience": 2,
  "Research Experience Depth": 5
}}

Respond ONLY with the JSON object, no additional text or explanation."""
        return prompt
    
    def extract_scores_from_response(self, response: str, category_batch: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, int]:
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                scores = json.loads(json_str)
            else:
                scores = json.loads(response)
            
            validated_scores = {}
            for cat_name, _ in category_batch:
                score = scores.get(cat_name)
                if score is not None:
                    try:
                        score_int = int(score)
                        if 0 <= score_int <= 5:
                            validated_scores[cat_name] = score_int
                        else:
                            print(f"Warning: Score {score_int} for {cat_name} out of range (0-5). Using 0.")
                            validated_scores[cat_name] = 0
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid score '{score}' for {cat_name}. Using 0.")
                        validated_scores[cat_name] = 0
                else:
                    print(f"Warning: Missing score for {cat_name}. Using 0.")
                    validated_scores[cat_name] = 0
            
            return validated_scores
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {response[:200]}...")
            
            # Fallback regex extraction
            scores = {}
            for cat_name, _ in category_batch:
                pattern1 = rf'"{re.escape(cat_name)}"\s*:\s*(\d)'
                pattern2 = rf'{re.escape(cat_name)}\s*:\s*(\d)'
                
                match1 = re.search(pattern1, response)
                match2 = re.search(pattern2, response)
                
                if match1:
                    score = int(match1.group(1))
                elif match2:
                    score = int(match2.group(1))
                else:
                    score = 0
                
                scores[cat_name] = score if 0 <= score <= 5 else 0
            
            return scores
        
        except Exception as e:
            print(f"Error extracting scores: {e}")
            return {cat_name: 0 for cat_name, _ in category_batch}
    
    def batch_categories(self, selected_categories: List[Tuple[str, Dict[str, Any]]], 
                    max_tokens: int = 3000) -> List[List[Tuple[str, Dict[str, Any]]]]:
        batches = []
        current_batch = []
        current_tokens = 0
        
        base_prompt = self.create_batched_prompt("", [])
        base_tokens = self.estimate_tokens(base_prompt)
        
        for cat_name, cat_info in selected_categories:
            cat_text = f"{cat_name}: {cat_info.get('question', '')}"
            cat_tokens = self.estimate_tokens(cat_text)
            
            # ✅ ENFORCE BOTH CONDITIONS:
            if ((current_tokens + cat_tokens + base_tokens > max_tokens) or 
                (len(current_batch) >= self.batch_size)) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append((cat_name, cat_info))
            current_tokens += cat_tokens
        
        if current_batch:
            batches.append(current_batch)
        
        print(f"Grouped {len(selected_categories)} categories into {len(batches)} batches")
        return batches
    
    def evaluate_document_batch(self, document_name: str, document_content: str, 
                           category_batches: List[List[Tuple[str, Dict[str, Any]]]]) -> Dict[str, int]:
        all_scores = {}
        
        for batch_num, category_batch in enumerate(category_batches, 1):
            cache_key = f"{document_name}_{hash(str(category_batch))}"
            
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                batch_scores = self.cache[cache_key]
            else:
                system_prompt = """You are an expert document evaluator..."""
                user_prompt = self.create_batched_prompt(document_content, category_batch)
                
                batch_scores = None
                for attempt in range(2):  # Try up to 2 times
                    try:
                        response = self.inferer.forward(system_prompt, user_prompt)
                        self.stats['total_calls'] += 1
                        batch_scores = self.extract_scores_from_response(response, category_batch)
                        
                        # Check completeness
                        missing = [cat for cat, _ in category_batch if cat not in batch_scores or batch_scores[cat] == 0]
                        if len(missing) > len(category_batch) * 0.5 and attempt == 0:
                            print(f"⚠️  Too many missing scores ({len(missing)}/{len(category_batch)}). Retrying...")
                            time.sleep(1)
                            continue  # Retry
                        break
                        
                    except Exception as e:
                        print(f"Error evaluating {document_name} batch {batch_num}: {e}")
                        self.stats['errors'] += 1
                        batch_scores = {cat_name: 0 for cat_name, _ in category_batch}
                        break
                
                if batch_scores is None:
                    batch_scores = {cat_name: 0 for cat_name, _ in category_batch}
                
                self.cache[cache_key] = batch_scores
                time.sleep(0.5)
            
            all_scores.update(batch_scores)
    
        return all_scores

    def write_header_if_needed(self, output_path: str, category_names: List[str]):
        """Write CSV header if file doesn't exist."""
        if not os.path.exists(output_path):
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['document_name'] + category_names)
    
    def append_result_to_csv(self, output_path: str, doc_name: str, scores: Dict[str, int], category_names: List[str]):
        """Append a single result to the CSV file."""
        row = [doc_name] + [scores.get(cat_name, 0) for cat_name in category_names]
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    def evaluate_documents_sequential(self, documents: Dict[str, str], 
                                     selected_categories: List[Tuple[str, Dict[str, Any]]],
                                     output_path: str) -> int:
        """Evaluate documents sequentially with resume support. Returns number of newly processed docs."""
        category_names = [cat[0] for cat in selected_categories]
        completed_docs = self.get_completed_documents(output_path)
        docs_to_process = {name: content for name, content in documents.items() if name not in completed_docs}
        
        if not docs_to_process:
            print("All documents already processed. Nothing to do.")
            return 0

        # Write header if needed
        self.write_header_if_needed(output_path, category_names)
        
        self.stats['start_time'] = time.time()
        self.stats['total_documents'] = len(docs_to_process)
        self.stats['total_categories'] = len(selected_categories)
        category_batches = self.batch_categories(selected_categories)
        
        print(f"\nStarting evaluation of {len(docs_to_process)} new documents...")
        print(f"Categories: {len(selected_categories)} grouped into {len(category_batches)} batches")
        print(f"Total API calls needed: {len(docs_to_process) * len(category_batches):,}")

        completed = 0
        newly_completed = 0
        
        for doc_name, doc_content in docs_to_process.items():
            try:
                scores = self.evaluate_document_batch(doc_name, doc_content, category_batches)
                self.append_result_to_csv(output_path, doc_name, scores, category_names)
                newly_completed += 1
            except Exception as e:
                print(f"Critical error processing {doc_name}: {e}")
                self.stats['errors'] += 1
                # Still write a row with zeros to avoid reprocessing
                fallback_scores = {cat_name: 0 for cat_name in category_names}
                self.append_result_to_csv(output_path, doc_name, fallback_scores, category_names)
            
            completed += 1
            self.stats['completed_docs'] = completed
            if completed % 5 == 0 or completed == len(docs_to_process):
                elapsed = time.time() - self.stats['start_time']
                docs_per_sec = completed / elapsed if elapsed > 0 else 0
                remaining = len(docs_to_process) - completed
                eta = remaining / docs_per_sec if docs_per_sec > 0 else 0
                
                print(f"  Progress: {completed}/{len(docs_to_process)} "
                      f"({completed/len(docs_to_process)*100:.1f}%) | "
                      f"Speed: {docs_per_sec:.2f} docs/sec | "
                      f"ETA: {str(timedelta(seconds=int(eta)))}")
                print(f"  API calls: {self.stats['total_calls']:,} | "
                      f"Cache hits: {self.stats['cache_hits']:,} | "
                      f"Errors: {self.stats['errors']}")

        return newly_completed

    def print_final_stats(self, total_original_docs: int, newly_completed: int, output_path: str):
        """Print final statistics including resume info."""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        total_processed = total_original_docs
        print(f"\nResults saved incrementally to {output_path}")
        print(f"Total documents in folder: {total_processed}")
        print(f"Newly processed in this run: {newly_completed}")
        print(f"Categories used: {self.stats['total_categories']}")
        
        if elapsed > 0:
            print(f"\n=== EVALUATION STATISTICS ===")
            print(f"Total time: {timedelta(seconds=int(elapsed))}")
            print(f"Documents processed this run: {newly_completed}")
            print(f"Categories evaluated: {self.stats['total_categories']}")
            print(f"Total API calls: {self.stats['total_calls']:,}")
            print(f"Cache hits: {self.stats['cache_hits']:,}")
            print(f"Errors: {self.stats['errors']}")
            print(f"Documents per second: {newly_completed / elapsed:.2f}")
            print(f"API calls per document: {self.stats['total_calls'] / max(newly_completed, 1):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Batched Document Evaluation Script with Resume Support")
    parser.add_argument("--categories", required=True, help="Path to categories JSON file")
    parser.add_argument("--folder", required=True, help="Folder containing .txt documents")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    parser.add_argument("--n", type=int, required=True, help="Number of top categories to use")
    parser.add_argument("--api_key", required=True, help="OpenRouter API key")
    parser.add_argument("--model", required=True, help="Model name (e.g., openai/gpt-3.5-turbo)")
    parser.add_argument("--batch_size", type=int, default=20, help="Max categories per API call (default: 20)")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.categories):
        print(f"Error: Categories file not found: {args.categories}")
        sys.exit(1)
    if not os.path.isdir(args.folder):
        print(f"Error: Document folder not found: {args.folder}")
        sys.exit(1)

    # Initialize evaluator
    evaluator = DocumentEvaluator(
        api_key=args.api_key,
        model=args.model,
        batch_size=args.batch_size
    )

    # Load data
    categories = evaluator.load_categories(args.categories)
    selected_categories = evaluator.select_top_categories(categories, args.n)
    all_documents = evaluator.read_documents(args.folder)

    if not all_documents:
        print("No documents found. Exiting.")
        sys.exit(1)

    # Evaluate (with resume support)
    newly_completed = evaluator.evaluate_documents_sequential(all_documents, selected_categories, args.output)

    # Print final stats
    evaluator.print_final_stats(len(all_documents), newly_completed, args.output)

    if newly_completed == 0:
        print("\n✅ No new documents processed (all were already done).")
    else:
        print(f"\n✅ Successfully processed {newly_completed} new documents.")


if __name__ == "__main__":
    main()