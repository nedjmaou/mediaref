import os
from tqdm.notebook import tqdm
import argparse
import pandas as pd

from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk import word_tokenize
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

# Calculate word-based metrics for single MBC
# (Adapted from Schlichtkrull 2024; ROUGE-1 and ROUGE-2 metrics added)
def evaluate_summaries(prediction, gold):
    # Calculate METEOR
    meteor = meteor_score([word_tokenize(gold)], word_tokenize(prediction))

    # Calculate ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, gold)

    rouge1 = scores['rouge1'].fmeasure
    rouge2 = scores['rouge2'].fmeasure
    rougeL = scores['rougeL'].fmeasure

    # Calculate token count
    token_count = len(word_tokenize(prediction))

    return {
        "METEOR": meteor,
        "ROUGE-1": rouge1,
        "ROUGE-2": rouge2,
        "ROUGE-L": rougeL,
        "token_length": token_count,
    }

# Calculate metrics for all MBCs
def process_files_metrics(predictions_folder, mbc_list_file, gold_folder, results_file, process_init=False):
    # Load filenames for generated MBCs
    with open(mbc_list_file, 'r') as f:
        dataset_lines = [line.split("\t")[0].strip() for line in f.readlines()]

    global_eval_results = {
        "METEOR": 0,
        "ROUGE-L": 0,
        "ROUGE-1": 0,
        "ROUGE-2": 0,
        "token_length": 0,
    }

    # Iterate through each text file in the folder
    list_results = []
    for filename in tqdm(dataset_lines):
        if process_init:
            prediction_file_path = os.path.join(predictions_folder, 'init_'+filename)
        else:
            prediction_file_path = os.path.join(predictions_folder, filename)
        gold_file_path = os.path.join(gold_folder, filename)

        # Read generated MBC text
        with open(prediction_file_path, 'r') as file:
            prediction = file.read()

        # Read gold-reference MBC text
        with open(gold_file_path, 'r') as file:
            gold = file.read()

        # Calculate metrics and add to cumulative total
        eval_results = evaluate_summaries(prediction, gold)
        for key in global_eval_results:
            global_eval_results[key] += eval_results[key]

        # Append results to list, to save to file
        list_results.append({'news_source' : filename.removesuffix('.txt')}|eval_results)

    # Find averages from cumulative totals, and print results
    for key in global_eval_results:
        global_eval_results[key] /= len(dataset_lines)

    for key, val in global_eval_results.items():
      print(f'{key}: {val:.4f}')

    # Save scores as .csv file
    df_results = pd.DataFrame(list_results)
    df_results.to_csv(results_file)

# Run evaluation on full predictions
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate MBCs using METEOR and ROUGE')
    parser.add_argument('--mbc_list_file', type=str, default='data/mbc_list')
    parser.add_argument('--predictions_folder', type=str, default='generated_mbcs')
    parser.add_argument('--gold_folder', type=str, default='data/gold_mbcs')
    parser.add_argument('--results_file', type=str, default='MBC_metric_results.csv')
    parser.add_argument('--process_init', type=bool, default=False) # True for MBCs without info retrieval; False for MBCs with info retrieval
    args = parser.parse_args()

    process_files_metrics(
        predictions_folder=args.predictions_folder,
        mbc_list_file=args.mbc_list_file,
        gold_folder=args.gold_folder,
        results_file=args.results_file,
        process_init=args.process_init
    )