from openai import OpenAI
import os
import tqdm
import time
import sys
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from spacy.lang.en import English
import json
import argparse
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

# Adapted from Schlichtkrull (2024).
class MBCGenerator:
    model_name = "deepset/deberta-v3-large-squad2" # Question-answering model
    qa_model = pipeline('question-answering', model=model_name, tokenizer=model_name)
    nlp = English()
    nlp.add_pipe("sentencizer") # Sentence segmentation, without loading dependancy parser
    local_model = False

    ## Constructor – Set up text generation model
    def __init__(self, model_name, knowledge_store_folder, local_model=False) -> None:
        # Define LLM model; load access keys from file
        self.model = model_name
        with open("config.json", "r") as f:
            config = json.load(f)
        openrouter_key = config["openrouter_key"]
        # openai_key = config["openai_key"]
        hf_access_token = config["hf_access_token"]

        # Create LLM client
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_key
        )

        # Import, index knowledge store with BM25.
        self.docs = []
        self.urls = []
        doc_urlnames = os.listdir(knowledge_store_folder)
        for f_name in doc_urlnames:
          with open(os.path.join(knowledge_store_folder, f_name), 'r') as f:
            f_text = f.read()
            if len(f_text) > 0:
              self.docs.append(f_text)
              self.urls.append(f_name)

        # Index documents for BM25 retrieval
        tokenised_chunks = []
        print('Tokenising documents...')
        for chunk in tqdm.notebook.tqdm(self.docs):
          tokenised_chunks.append(self.get_word_tokens(chunk))
        print('Indexing documents...')
        self.bm25 = BM25Okapi(tokenised_chunks)
        print('Knowledge store BM25-indexed.')

        # Use open-source model, if set (unused for this project)
        if local_model:
            self.local_model = True
            self.local_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
            access_token = hf_access_token
            self.local_tokenizer = AutoTokenizer.from_pretrained(self.local_model_id, token=access_token)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                self.local_model_id,
                #torch_dtype=torch.bfloat16,
                #device_map="cpu",
                token=access_token
                )
        else:
            self.local_model = False

    # Generate response from local text-generation model
    def send_message_local(self, message):
        input_ids = self.local_tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt"
            ).to(self.local_model.device)

        terminators = [
            self.local_tokenizer.eos_token_id,
            self.local_tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

        outputs = self.local_model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            )

        response = outputs[0][input_ids.shape[-1]:]
        response = self.local_tokenizer.decode(response, skip_special_tokens=True)
        return response

    # Generate response from GPT3.5 model given prompt, if using.
    # Else, generate using the local model.
    def send_message(self, message):
        if self.local_model:
            return self.send_message_local(message)

        result = None
        retries = 0
        while result is None:
            try:
                # Prompt the GPT model
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message
                )
                result = response.choices[0].message.content
            except:
                # If prompt fails, retry after exponential wait time
                print(f"Failed to send message. Retrying (attempt {retries + 1})...", file=sys.stderr)
                delay = min(1 * (2 ** retries), 64)
                time.sleep(delay)
                retries += 1

        return result

    # Given question and single retrieved piece of evidence,
    # generate an answer using the question-answering model.
    def get_answer_from_evidence(self, question, evidence, qa_model=qa_model, nlp=nlp):
        evidence = str(evidence)[:3000].strip()
        if len(evidence) == 0:
            return {
                "backing": "none",
                "answer": "unavailable",
            }

        # Set up input for QA model
        QA_input = {
            'question': question,
            'context': evidence
        }

        # Get response from QA model
        response = qa_model(**QA_input)

        # If model is uncertain, return no answer
        if response["score"] < 0.2:
            return {
                "backing": "none",
                "answer": "unavailable",
            }
        answer = response["answer"]
        start_token = response["start"]
        end_token = response["end"]

        # Gather `backing' sections from text supporting answer
        backing = []
        sections_to_retrieve = 0
        for section in nlp(evidence).sents:
            section_start = section.start_char
            section_end = section.end_char

            if start_token >= section_start:
                sections_to_retrieve = 1
            if sections_to_retrieve:
                backing.append(section.text)
            if end_token <= section_end:
                sections_to_retrieve = 0
        backing = " ".join(backing)

        if len(backing) == 0:
            print("WARNING: I failed at producing JSON", file=sys.stderr)
            return {
                "backing": "none",
                "answer": "unavailable",
            }

        # Convert answer and backing to json; test if this is successful
        result = json.dumps({
            "backing":backing,
            "answer": answer
        })
        try:
            result = json.loads(result)
        except:
            print("WARNING: I failed at producing JSON", file=sys.stderr)
            return {
                "backing": "none",
                "answer": "unavailable",
            }

        # If we have no answer, or the answer is unavailable, we just stop here. TODO: We probably want to do some double checking.
        if "answer" not in result:
            result["answer"] = "unavailable"
        if result["answer"] == "unavailable":
            return result

        # Verify that the answer contains backing, and that the backing is actually in the original document:
        if "backing" not in result:
            result["answer"] = "unavailable"

        return result

    # Given a query, retrieve the top n documents from knowledge store with BM25
    def retrieve_docs(self, query, n=30):
        tokenised_query = word_tokenize(query)
        retrieved_doc_text = self.bm25.get_top_n(tokenised_query, self.docs, n)
        retrieved_doc_links = [self.urls[self.docs.index(doc)] for doc in retrieved_doc_text]

        # Return documents and associated URLs
        results = [{'evidence' : text, 'link' : link}
                   for text, link in zip(retrieved_doc_text, retrieved_doc_links)]
        return results

    # Generate initial MBC for news source, without additional retrieved info
    def generate_initial_guess(self, source, use_icl=False):
        # Define LLM prompt text
        system_message = "You are InfoHuntGPT, a world-class AI assistant used by journalists to quickly build knowledge of new sources."
        prompt = "Build a background check for the news source \"" + source + "\". Write down everything you know about them, e.g. who funds them, how they make money, if they have any particular bias. Make an ITEMIZED LIST. Be brief, and if you don't know something, just leave it out.\n"
        prompt += "If you are aware that they have failed any fact-checks, mention which. Begin your response with \"**Background check**\"."
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        # Return LLM response
        result = self.send_message(messages)
        return result

    # Incorporate additional retrieved information into an existing MBC
    def incorporate_extra_information(self, source, previous_guess, new_information):
        system_message = "You are InfoHuntGPT, a world-class tool used by journalists to quickly build knowledge of new sources."
        initial_prompt = "Build a background check for the news source \"" + source + "\". Write down everything you know about them, e.g. who funds them, how they make money, if they have any particular bias. Make an ITEMIZED LIST. Be brief, and if you don't know something, just leave it out.\n"
        initial_prompt += "If you are aware that they have failed any fact-checks, mention which. Begin your response with \"**Background check**\"."
        bot_answer = previous_guess
        prompt = "Google search has revealed some new information:\n\n"
        prompt += new_information + "\n\n"
        prompt += "Update your background check for \"" + source + "\" using the new information. Do NOT delete any information, but make ADDITIONS where necessary, using the new information. Most likely, you will just need to add an extra item to the itemized list you previously created. Make minimal edits, and only incorporate what is relevant. Begin your response with \"**Background check**\""
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": bot_answer},
            {"role": "user", "content": prompt},
        ]

        # Return LLM response
        result = self.send_message(messages)
        return result

    # Build MBC for given source, with information retrieval over all fact-finding queries
    def build_background_check(self, source, questions_file="queries.json", run_search=True, back_initial=True):
        # Load list of queries
        with open(questions_file, 'r') as file:
            queries = json.load(file)

        # Generate initial MBC
        initial_guess = self.generate_initial_guess(source)

        # If only generating initial MBC, stop here
        bgcheck = initial_guess
        if not run_search:
            return bgcheck, None

        # For each query, retrieve info and incorporate into MBC
        search_lines = []
        for item in tqdm.notebook.tqdm(queries, leave=False):
            localized_question = item['question'].strip().replace("X", "\"" + source + "\"")
            localized_search_query = item['statement'].strip().replace("X", "\"" + source + "\"")

            # Retrieve n search results from knowledge store
            search_results = self.retrieve_docs(localized_search_query, n=30) # default n = 30

            # For each retrieved result, extract an answer with DeBERTa QA model
            extra_info = ""
            for s in tqdm.notebook.tqdm(search_results, leave=False):
                s["question"] = localized_search_query
                search_lines.append(s)
                evidence = s["evidence"]

                # If no answer found, skip
                answer = self.get_answer_from_evidence(localized_question, evidence)
                if answer["answer"] == "unavailable":
                    # print('No answer found')
                    continue

                # If answer found, keep corresponding backing
                if len(extra_info) > 0:
                    extra_info += "\n"
                extra_info += localized_question.replace("\"", "") + " " + answer["backing"]

            # Update MBC with all backing found for query
            if len(extra_info) > 0:
                bgcheck = self.incorporate_extra_information(source, bgcheck, extra_info)

        # Finally, update MBC with initial guess info (to reincorporate overwritten info)
        if back_initial:
            bgcheck = self.incorporate_extra_information(source, bgcheck, initial_guess)
        return bgcheck, initial_guess, search_lines

# Loop through given filenames of sources, and generate an MBC for each.
def process_tsv(model_name, input_file, output_folder, knowledge_store_folder, search=True, local_model=False):
    mbc_generator = MBCGenerator(
        model_name=model_name,
        knowledge_store_folder=knowledge_store_folder,
        local_model=local_model
    )
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Read the file
    with open(input_file, 'r') as f:
        lines = [line.strip().split('\t') for line in f.readlines()]

    # Iterate through each line
    pbar = tqdm.notebook.tqdm(lines, desc="Processing files")
    for line in pbar:
        # Get source name, set output file name
        test_source = line[0]  # Get the name from the first column
        pbar.set_description(f"Processing \"{test_source}\"")
        output_file = os.path.join(output_folder, test_source)
        output_file_init = os.path.join(output_folder, 'init_'+test_source)

        # If MBC for source already exists, skip this source
        if os.path.exists(output_file):
          continue

        # Generate MBC for source
        source_check, initial_guess, search_lines = mbc_generator.build_background_check(test_source.replace(".txt", ""), run_search=search)

        # Save MBCs both with and without incorporated information to file
        with open(output_file, 'w') as f:
            f.write(source_check)
        with open(output_file_init, 'w') as f:
            f.write(initial_guess)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate media background checks')
    parser.add_argument('--model_name', type=str, default='meta-llama/llama-3.3-70b-instruct')
    parser.add_argument('--input_file', type=str, default='data/mbc_list.tsv')
    parser.add_argument('--output_folder', type=str, default='/generated_mbcs')
    parser.add_argument('--knowledge_store_folder', type=str, default='/store_texts')
    args = parser.parse_args()

    process_tsv(
        model_name=args.model_name,
        input_file=args.input_file,
        output_folder=args.output_folder,
        knowledge_store_folder=args.knowledge_store_folder,
        search=True,
        local_model=False
    )