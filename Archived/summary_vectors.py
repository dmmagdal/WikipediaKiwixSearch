# summary_vectors.py
# Iterate through all (valid) articles and embed the title + summary 
# (first paragraph) for each one. The idea is to perform a semantic
# search on the article summaries rather than the entire article for
# faster article sorting.
# Python 3.9
# Windows/MacOS/Linux


import os

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
		for entry_id in tqdm(entry_ids[:100]):
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

			# # Given the HTML, extract the 
			# content_div = soup.find("div", {"id": "mw-content-text"})
			# if not content_div:
			# 	# return ""
			# 	continue

			# lead_paras = []
			# for child in content_div.children:
			# 	# if getattr(child, "name", None) == "h2":
			# 	# 	break  # stop at first heading

			# 	if getattr(child, "name", None) == "p":
			# 		text = child.get_text(strip=True)
			# 		if text:
			# 			lead_paras.append(text)
			# 			break

			# # return " ".join(lead_paras)
			# summary = " ".join(lead_paras)

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
			start_idx = article_lines.index(entry.title) + 1
			# print(start_idx)
			# print(len(article_lines))

			# Iterate from the starting index through the length of the
			# article.
			for idx in range(start_idx, len(article_lines)):
				# print(idx)

				# Isolate the line (strip whitespace).
				line = article_lines[idx].strip()
				# print(line)

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

			print()
			print()
			print(entry.title)
			print(url)
			# print(article_text)
			print(first_paragraph)
			print("-" * 72)
			# print(item_html_text)
			# print(summary)
			# exit()

			# NOTE:
			# This has only been validated on parts of Simple English 
			# Wikipedia. Would recommend running this on full English
			# Wikipedia to make sure the first paragraphs are of 
			# sufficient quality to summarize the article.

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()