from openai import OpenAI
import os
from tqdm.notebook import tqdm
import time
import sys
from collections import Counter
import json
import argparse
import pandas as pd
from sentence_transformers import CrossEncoder

# Read config file for OpenAI key:
with open("config.json", "r") as f:
    config = json.load(f)
    openrouter_key = config["openrouter_key"]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key
)

# Define helper functions
def most_frequent_element(input_list):
    return Counter(input_list).most_common(1)[0][0] if input_list else None

def send_message(message):
    model = "gpt-3.5-turbo"
    response = client.chat.completions.create(
        model=model,
        messages=message
    )
    text = response.choices[0].message.content
    return text

# Define the DeBERTa model used for checking entailment
model = CrossEncoder('cross-encoder/nli-deberta-v3-large')
# Predict entailment of a claim given text, using DeBERTa
def check_implication_deberta(claim, text):
  scores = model.predict([(text, claim)])

  label_mapping = ['contradiction', 'entailment', 'neutral']
  out_label = label_mapping[scores.argmax(axis=1)[0]]
  return out_label

# Predict entailment of a claim given text, using GPT-3.5
def check_implication(claim, text, passes=4):
    votes = [gpt_check_implication(claim, text) for _ in range(passes)]

    majority = most_frequent_element(votes)
    return majority

def gpt_check_implication(question, text):
    # Define prompts
    system_message = "You are FactCheckGPT, a world-class tool used by journalists to discover problems in their writings. Users give you text, and check whether facts are true given the text. You ALWAYS answer either TRUE, FALSE, or NOT ENOUGH EVIDENCE."
    prompt = "You will be given a snippet written as part of a source criticism exercise, and a claim. Your task is to determine whether the claim is true based ONLY on the text. Do NOT use any other knowledge source\n\n"
    prompt += "The claim is: \"" + question + "\".\n"
    prompt += "The text follows below:\n\"" + text + "\".\n\n"
    prompt += question + " Thinking step by step, answer either TRUE, FALSE, or NOT ENOUGH EVIDENCE, capitalizing all letters. Explain your reasoning FIRST, and after that output either TRUE, FALSE, or NOT ENOUGH EVIDENCE."

    # Construct chat messages
    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

    # Obtain chat response; if no response, try again with exponential wait times
    retries = 0
    result = None
    while result is None:
        try:
            result = send_message(messages)
        except Exception as e:
            print(f"Failed to send message. Retrying (attempt {retries + 1})...", file=sys.stderr)
            print(f"Exception: {e}")
            delay = min(1 * (2 ** retries), 64)
            time.sleep(delay)
            retries += 1

    # Output entailment verdict
    if "TRUE" in result:
        return "entailment"
    elif "FALSE" in result:
        return "contradiction"
    else:
        return "neutral"

# Calculate atomic fact–based metrics for single MBC
def process_files_atomic_facts(predictions_folder, mbc_list_file, fact_folder, results_filename, start_at=0, end_at=-1, entailment_method='gpt', passes=4, process_init=False):
    # Read filenames of gold and generated MBCs
    with open(mbc_list_file, 'r') as f:
        if end_at == -1:
            gold_dataset_lines = [line.split("\t")[0].strip() for line in f.readlines()][start_at:]
        else:
            gold_dataset_lines = [line.split("\t")[0].strip() for line in f.readlines()][start_at:end_at]
    if process_init:
        pred_dataset_lines = ['init_' + line for line in gold_dataset_lines]
    else:
        pred_dataset_lines = gold_dataset_lines

    recalled_info_rate = 0
    error_rate = 0

    fact_eval_results = []
    source_eval_results = []
    # Iterate through each MBC filename
    for gold_filename, pred_filename in tqdm(zip(gold_dataset_lines, pred_dataset_lines), total=len(gold_dataset_lines)):
        this_file_info_rate = 0
        this_file_error_rate = 0
        if True:
            # get full filepath for predicted, gold MBCs
            prediction_file_path = os.path.join(predictions_folder, pred_filename)
            fact_file_path = os.path.join(fact_folder, gold_filename)
            score = {
                "entailment": {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0
                },
                "contradiction": {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0
                },
                "neutral": {
                    "fp": 0,
                }
            }

            # Read predicted and gold MBC files
            with open(prediction_file_path, 'r') as file:
                prediction = file.read()
            with open(fact_file_path, 'r') as file:
                facts = [f.strip().split("\t") for f in file.read().split("\n") if f.strip()]

            # Iterate through each fact in gold MBC, and check entailment on predicted MBC
            for fact in tqdm(facts, leave=False):
                if entailment_method == 'deberta':
                    entail_pred = check_implication_deberta(fact[0], prediction)
                else:
                    entail_pred = check_implication(fact[0], prediction, passes=passes)
                entail_gold = fact[1]

                # Record entailment verdicts
                fact_eval_results.append({
                    'source' : gold_filename.removesuffix('.txt'),
                    'fact' : fact[0],
                    'predicted' : entail_pred,
                    'gold' : entail_gold
                })

                # Record counts of false-positives, false-negatives and true-positives
                if entail_pred != entail_gold:
                    score[entail_pred]["fp"] += 1
                    score[entail_gold]["fn"] += 1
                else:
                    score[entail_pred]["tp"] += 1

            # Calculate fact recall and error rates
            if len(facts) == 0:
                recalled_info_rate += 1.0
                this_file_info_rate += 1.0
                this_file_error_rate = 0.0
            else:
                recalled_info_rate +=  (score["entailment"]["tp"] + score["contradiction"]["tp"]) / len(facts)
                error_rate += (score["entailment"]["fp"] + score["contradiction"]["fp"]) / len(facts)

                this_file_info_rate = (score["entailment"]["tp"] + score["contradiction"]["tp"]) / len(facts)
                this_file_error_rate = (score["entailment"]["fp"] + score["contradiction"]["fp"]) / len(facts)

            # Record metrics for each source
            source_eval_results.append({
                'source' : gold_filename.removesuffix('.txt'),
                'info_rate' : this_file_info_rate,
                'error_rate' : this_file_error_rate
            })

    # If reading entire dataset, print averages
    if end_at == -1:
        print(f'Cumulative recalled info rate: {recalled_info_rate:.4}')
        print(f'Cumulative error rate: {error_rate:.4}')
        print(f'Av. recalled info rate: {recalled_info_rate / len(gold_dataset_lines):.2%}')
        print(f'Av. error rate: {error_rate / len(gold_dataset_lines):.2%}')

    # Save results to csv files
    df_fact_eval = pd.DataFrame(fact_eval_results)
    df_source_eval = pd.DataFrame(source_eval_results)
    df_fact_eval.to_csv(results_filename+'_facts.csv')
    df_source_eval.to_csv(results_filename+'sources.csv')

    # Return entailment results for facts, and metrics for individual sources
    return (fact_eval_results, source_eval_results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MBCs using atomic facts')
    parser.add_argument('--start_at', type=int, default=0)
    parser.add_argument('--end_at', type=int, default=-1)
    parser.add_argument('--mbc_list_file', type=str, default='data/mbc_list')
    parser.add_argument('--predictions_folder', type=str, default='generated_mbcs')
    parser.add_argument('--fact_folder', type=str, default='data/eval_facts')
    parser.add_argument('--results_file', type=str, default='MBC_atomic_fact_results.csv')
    parser.add_argument('--process_init', type=bool, default=False) # True for MBCs without info retrieval; False for MBCs with info retrieval
    args = parser.parse_args()

    fact_eval_results, source_eval_results = process_files_atomic_facts(
        predictions_folder=args.predictions_folder,
        mbc_list_file=args.mbc_list_file,
        fact_folder=args.fact_folder,
        start_at=args.start_at,
        end_at=args.end_at,
        process_init=args.process_init,
        entailment_method='gpt3-5',
        passes=1
    )