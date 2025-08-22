# random_wiki.py
# Randomly get articles from the wikipedia (online) via the 
# wikipedia-api. 
# Python 3.9
# Windows/MacOS/Linux


import requests
import wikipediaapi


def main():
	# Initialize wikipedia-api
	wiki = wikipediaapi.Wikipedia(
		user_agent="WikiHybridRAG/1.0 (https://example.com; contact@example.com)",
		language="en"
	)

	response = requests.get(
		"https://en.wikipedia.org/w/api.php",
		params={
			"action": "query",
			"list": "random",
			"rnnamespace": 0,  # namespace 0 = articles only
			"rnlimit": 1,
			"format": "json"
		}
	)
	data = response.json()
	title = data["query"]["random"][0]["title"]

	# Now fetch the page content using wikipedia-api
	page = wiki.page(title)
	
	print(f"Page: {page.title}")
	print(f"{page.text}")

	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()