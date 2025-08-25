# precompute_dense_vectors.py
# Precompute all necessary dense vector representations. These vectors
# live in the same folder but not the same table in lancedb.
# Python 3.9
# Windows/MacOS/Linux


import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import multiprocessing as mp
import os
import time
from typing import List

from bs4 import BeautifulSoup
import lancedb
from libzim.reader import Archive
import pyarrow as pa
import torch
from tqdm import tqdm

from preprocess import vector_preprocessing, load_model


def get_attention_mask(tokens: List[int], pad_token_id: int) -> List[int]:
	return [0 if t == pad_token_id else 1 for t in tokens]


def main():
	# Initialize argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--num_proc",
		type=int,
		default=1,
		help="How many processors to use. Default is 1."
	)
	parser.add_argument(
		"--num_thread",
		type=int,
		default=1,
		help="How many threads to use. Default is 1."
	)
	parser.add_argument(
		"--restart",
		action="store_true",
        help="Whether to restart the embedding process from scratch. Default is false/not specified."
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="The size of the batches for the vector embedding model. Default is 1."
	)

	# Parse arguments.
	args = parser.parse_args()
	num_proc = args.num_proc
	num_thread = args.num_thread
	restart = args.restart
	batch_size = args.batch_size

	num_cpus = min(mp.cpu_count(), num_proc)
	max_workers = num_cpus if num_proc > 1 else num_thread

	# Load config file and isolate key variables.
	if not os.path.exists("config.json"):
		print("Could not detect required file config.json in current path.")
		print("Exiting program.")
		exit(1)

	with open("config.json", "r") as f:
		config = json.load(f)

	# Isolate paths.
	vector_search_config = config["vector-search_config"]
	db_uri = vector_search_config["db_uri"]
	summary_table = vector_search_config["summary_table"]

	# Load the tokenizer and model.
	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda"
	elif torch.backends.mps.is_available():
		device = "mps"

	tokenizer, model = load_model(config, device)

	# Initialize index directory if it doesn't already exist.
	if not os.path.exists(db_uri):
		os.makedirs(db_uri, exist_ok=True)

	data_folder = "./WikipediaData"
	data_zim_files = [
		os.path.join(data_folder, file)
		for file in os.listdir(data_folder)
		if file.endswith(".zim")
	]

	# Connect to database.
	db = lancedb.connect(db_uri)

	# Load model dims to pass along to the schema init.
	model_name = config["vector-search_config"]["model"]
	dims = config["models"][model_name]["dims"]

	# Initialize schema (this will be passed to the database when 
	# creating a new, empty table in the vector database).
	schema = pa.schema([
		pa.field("file", pa.utf8()),
		pa.field("entry_id", pa.int32()),
		pa.field("text_idx", pa.int32()),
		pa.field("text_len", pa.int32()),
		pa.field("vector", pa.list_(pa.float32(), dims))
	])

	# Refresh the table (drop it) if it exists but we're not updating 
	# it.
	current_tables = db.table_names()
	if summary_table in current_tables and restart:
		db.drop_table(summary_table)

	# Initialize the fresh table.
	if summary_table not in current_tables:
		db.create_table(summary_table, schema=schema)

	# Get the table for the file.
	table = db.open_table(summary_table)

	# NOTE:
	# I'd love for this script to be multi-threaded/multi-processing
	# but lancedb has gotten weird when I try and pass the connection
	# (via reference) to functions wrapped within thread/process pools.
	# Until I can figure out a way to do that properly (and safely) 
	# this process will have to be bottlenecked by single-threaded/
	# single-process execution.
	# - I have considered the idea of keeping a buffer of the metadata
	# generated from the embedding processs/function and then uploading
	# that buffer to the database every so many steps, but that very 
	# quickly becomes an over engineered mess just to eek out some 
	# efficiency and get bottlenecked by the local pool executor.

	# Iterate through each file.
	for idx, file in enumerate(data_zim_files):
		# Isolate the basename and print out the current file and
		# its position.
		basename = os.path.basename(file)
		print(f"Processing {basename} ({idx + 1}/{len(data_zim_files)})")

		# Load the file and get the list of entry items.
		archive = Archive(file)
		entry_ids = [i for i in range(archive.article_count)]

		# Identify any existing entry IDs that have already been
		# catalogued (if we are not restarting from scratch),
		if not restart:
			# Table query (export to pandas dataframe and work from 
			# there).
			table_df = table.to_pandas()
			existing_ids = table_df[
				(table_df["file"] == file) & (table_df["entry_id"].isin(entry_ids))
			]["entry_id"].tolist()

			# Remove the found existing entry IDs from the list.
			entry_ids = list(set(entry_ids) - set(existing_ids))

		# Initialize an empty batch buffer.
		batch = []

		# Iterate over the entry IDs.
		for entry_id in tqdm(entry_ids):
			# Isolate the entry and path/url.
			entry = archive._get_entry_by_id(entry_id)
			title = entry.title
			url = entry.path

			# Skip if the article is a redirect or asset.
			if entry.is_redirect or url == "null":
				continue

			###########################################################
			# Get the summary (first paragraph) of the article.
			###########################################################
			# Get the article text.
			item = entry.get_item()
			item_html_text = bytes(item.content).decode(
				"utf-8", errors="ignore"
			)
			soup = BeautifulSoup(item_html_text, "lxml")

			# Identified the first paragraph of an article via 
			# classical means (rather than HTML parsing).
			article_text = soup.text

			# Initialize first paragraph variable to be an empty 
			# string.
			first_paragraph = ""

			# Split the articles (by newline characters) and identify
			# the first index where the article title is present 
			# (increment that by 1 to be the starting index for the 
			# below for-loop).
			article_lines = article_text.splitlines()
			start_idx = article_lines.index(title) + 1 if title in article_lines else 0

			# Iterate from the starting index through the length of the
			# article.
			for idx in range(start_idx, len(article_lines)):
				# Isolate the line (strip whitespace).
				line = article_lines[idx].strip()

				# If the line is not empty and is not a match for the
				# article title, set the first paragraph to that line
				# and break the loop.
				if len(line) > 0 and line != entry.title:
					first_paragraph = line
					break

			# If the first paragraph is still an empty string, no 
			# paragraph was found, so set it to the article title.
			if first_paragraph == "":
				first_paragraph = entry.title

			# Tokenize.
			tokens = vector_preprocessing(
				first_paragraph, config, tokenizer, 
				recursive_split=False
			)

			# Append the metadata to the batch. Iterate because tokens
			# (returned by vector_preprocessing()) is a List[Dict]
			# containing the tokenized information.
			for token_set in tokens:
				batch.append((file, entry_id, first_paragraph, token_set))

			###########################################################
			# Embed the batch and store it to the vector db
			###########################################################
			if len(batch) >= batch_size:
				with torch.no_grad():
					# Embed the batch.
					batch_input_ids = torch.tensor(
						[
							article_tokens["tokens"] 
							for _, _, _, article_tokens in batch
						]
					).to(device)
					masks = torch.tensor(
						[
							get_attention_mask(
								article_tokens["tokens"], tokenizer.pad_token_id
							)
							for _, _, _, article_tokens in batch
						]
					).to(device)
					embeddings = model(
						input_ids=batch_input_ids, attention_mask=masks
					)

					# Compute the embedding by taking the mean of the
					# last hidden state tensor across the seq_len axis.
					embeddings = embeddings[0].mean(dim=1)

					# Apply the following transformations to allow the
					# embedding to be compatible with being stored in 
					# the vector DB (lancedb):
					#	1) Send the embedding to CPU (if it's not 
					#		already there)
					#	2) Convert the embedding to numpy and flatten 
					# 		the embedding to a 1D array
					if device != "cpu":
						embeddings = embeddings.to("cpu")

				# Convert embeddings from torch tensor to numpy array.
				embeddings = embeddings.numpy()

				# Recompile the metadata with the embeddings for chunk
				# writing to the table.
				metadata = []
				for idx, batch_data in enumerate(batch):
					file, entry_id, _, batch_tokens = batch_data
					embedding = embeddings[idx, :]
					metadata.append(
						{
							"file": file,
							"entry_id": entry_id,
							"text_idx": batch_tokens["text_idx"],
							"text_len": batch_tokens["text_len"],
							"vector": embedding
						}
					)
				table.add(metadata)

				# Clean up old manifest files (this can take up a lot
				# of space given the number of transactions/writes).
				table.optimize(
					cleanup_older_than=timedelta(seconds=30)
				)
				table.cleanup_old_versions(
					older_than=timedelta(seconds=30)
				)

				# Clear the batch buffer.
				batch = []

	# Clean up old manifest files (again).
	table.optimize(cleanup_older_than=timedelta(seconds=60))
	table.cleanup_old_versions(older_than=timedelta(seconds=60))

	# NOTE:
	# The table clean up helps cut back on storage. You can't just
	# remove the _versions or _transactions folders because that will
	# break the table. Highly recommend instilling this clean up on ALL
	# areas where tables are modified (add, update, delete, etc). Is not
	# necessary to do this for query (search) operations on the table.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()