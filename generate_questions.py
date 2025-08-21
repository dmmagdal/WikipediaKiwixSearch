# generate_questions.py
# Randomly select articles (and paragraphs) from the wikipedia corpus
# and use a small LLM to generate questions that can be answered by 
# that context passage. 
# Python 3.9
# Windows/MacOS/Linux


import math
import json
import os
import random
import shutil

from libzim.reader import Archive
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline

from searchwiki import get_all_articles, get_article_entries
from searchwiki import get_article_lengths, get_random_paragraph_from_article


# Set seed.
seed = 1234
random.seed(seed)
np.random.seed(seed)


def main():
	###################################################################
	# SETUP
	###################################################################
	# Paths for required files/folders.
	data_path = "./WikipediaData"
	config_path = "config.json"

	# Validate necessary paths.
	if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
		print(f"Error: Unable to find required data in {data_path}")
		print("Please run download script download.py to get the data.")
		exit(1)
	if not os.path.exists(config_path):
		print(f"Error: Required file config.json is not available.")
		print("Please re-download the repo to get the file.")
		exit(1)

	# Load config from file.
	with open (config_path, "r") as f:
		config = json.load(f)

	###################################################################
	# RANDOMLY SELECT ARTICLE/PASSAGE
	###################################################################
	# Randomly select articles from the corpus (weigh by article 
	# log-length).
	articles = get_all_articles(config)
	article_lengths = get_article_lengths(config, articles)
	article_lengths = [math.log(length) for length in article_lengths]
	probabilities = np.array(article_lengths) / np.sum(article_lengths)
	selected_docs = np.random.choice(
		articles, 
		size=5,
		replace=False,
		p=probabilities
	)

	# Select the query passages from the articles sampled.
	article_entries = get_article_entries(selected_docs)
	query_passages = [
		get_random_paragraph_from_article(text)
		for _, text, _, _ in article_entries
	]

	###################################################################
	# GENERATE QUESTION.
	###################################################################
	# Model IDs & their respective folders.
	models = ["google/flan-t5-large", "facebook/bart-large"]
	model_paths = [model_id.replace("/", "_") for model_id in models]
	model_caches = [model_path + "_tmp" for model_path in model_paths]

	# Device setup.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	# elif torch.backends.mps.is_available():
	# 	device = "mps"

	# NOTE:
	# Device setup for MPS (Apple Silicon) is finnicky for this 
	# environment. Recommend just using CPU since the models are 
	# relatively lightweight.

	# Build the output data structure.
	output = [
		{"article": article, "passage": text}
		for article, text, _, _ in article_entries
	]

	# Iterate through the models.
	for m_idx, model_id in enumerate(models):
		print(f"Running model: {model_id}")

		model_dir = model_paths[m_idx]
		model_cache = model_caches[m_idx]
		if not os.path.exists(model_dir):
			# Download the model and tokenizer.
			tokenizer = AutoTokenizer.from_pretrained(
				model_id, 
				cache_dir=model_cache
			)
			model = AutoModelForSeq2SeqLM.from_pretrained(
				model_id,
				cache_dir=model_cache
			)

			# Save the model and tokenizer.
			tokenizer.save_pretrained(model_dir)
			model.save_pretrained(model_dir)

			# Delete the cache folder from the download.
			shutil.rmtree(model_cache)
		
		# Load the model and tokenizer.
		tokenizer = AutoTokenizer.from_pretrained(model_dir)
		model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

		# Iterate through the passages.
		for idx, passage in tqdm(enumerate(query_passages)):
			# Build the prompt.
			prompt = f"Generate a question from this passage:\n{passage}"
			encoded_prompt = tokenizer.encode(
				prompt, add_special_tokens=False
			)

			# Skip the passage if it is too long (for the 
			# model/tokenizer).
			if len(encoded_prompt) > tokenizer.model_max_length:
				continue

			# Generate the question by passing the prompt to the model
			# pipeline.
			model_pipe = pipeline(
				"text2text-generation", 
				model=model,
				tokenizer=tokenizer, 
				device=device
			)
			model_output = model_pipe(
				prompt, 
				max_length=512, 
				# num_return_sequences=5	# Cannot use this if using greedy decoding (default params).
			)

			# Update the output.
			output[idx].update({model_id: model_output})

	# Save the output to JSON.
	with open("questions.json", "w+") as f:
		json.dump(output, f, indent=4)

	# NOTE:
	# It seems like Flan-T5 had the best generated questions. 
	# BART-large seemed to include the initial prompt and appened a 
	# random segment of the input passage/context after that.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()