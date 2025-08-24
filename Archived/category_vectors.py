# category_vectors.py
# Iterate through all (valid) articles and map out categories for each 
# one.
# Python 3.9
# Windows/MacOS/Linux


import os
from typing import Dict, List

from bs4 import BeautifulSoup
from libzim.reader import Archive
from tqdm import tqdm


def main():
	# XML files containing the actual documents.
	data_folder = "./WikipediaData"
	data_zim_files = [
		os.path.join(data_folder, file)
		for file in os.listdir(data_folder)
		if file.endswith(".zim")
	]
	
	# Iterate through each file.
	for idx, file in enumerate(data_zim_files):
		# Isolate the basename and print out the current file and
		# its position.
		basename = os.path.basename(file)
		print(f"Processing {basename} ({idx + 1}/{len(data_zim_files)})")

		# Load the file and get the list of entry ites.
		archive = Archive(file)
		entry_ids = [i for i in range(archive.article_count)]

		# Iterate through each article.
		for entry_id in tqdm(entry_ids[:50]):
			# Isolate the article/page's URL.
			entry = archive._get_entry_by_id(entry_id)
			url = entry.path

			# Skip articles that have a redirect tag (they have no 
			# useful information in them).
			if entry.is_redirect or url == "null":
				continue

			# Get the article text.
			item = entry.get_item()
			item_html_text = bytes(item.content).decode(
				"utf-8", errors="ignore"
			)
			soup = BeautifulSoup(item_html_text, "lxml")

			# print()
			# print()
			# print(entry.title)
			# print(url)
			# print(item_html_text)
			# exit()


			# cat_links = soup.select("div#catlinks ul li a")
			# categories = [a.get_text() for a in cat_links]
			# if len(categories) == 0:
			# 	continue
			# else:
			# 	print(url)
			# 	print(categories)
			# 	exit()


			# article_text = soup.text

			# if "category:" not in article_text.lower() or "categories:" not in article_text.lower():
			# 	continue
			# else:
			# 	print(article_text)
			# 	exit()

			# text = item.

			# NOTE:
			# Was NOT able to find categories (either in the HTML or 
			# the actual text) for all (valid) articles. This has 
			# pushed me to abandon this idea to use a stage 1 composed 
			# of query embeddings searching against category embeddings
			# (where each category maps to related articles).

	# Isolate all valid categories from (valid) articles.
	
	# Make sure each article has at least one category.
	
	# Embed the categories and store the metadata to a vector DB (i.e. 
	# the articles they belong to (entry ID).
	
	# Save the mapping (and inverse mapping) of articles to categories.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()