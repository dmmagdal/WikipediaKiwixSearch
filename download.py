# download.py
# Download parts of or all of the latest English wikipedia data from 
# their archive.
# Python 3.9
# Windows/MacOS/Linux


import argparse
import hashlib
import os

from bs4 import BeautifulSoup
import requests


def main():
	# Set up argument parser.
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--target", 
		nargs="?",
		default="all-simple-maxi", 
		help="Specify which (compressed) file(s) you want to download. Default is 'all-simple-maxi'."
	)
	parser.add_argument(
		"--no-shasum-check", 
		action="store_true",
		help="Whether to allow mismatched SHA256 on download to result in a failure. Default is false/not specified."
	)
	args = parser.parse_args()

	###################################################################
	# CONSTANTS
	###################################################################
	# Core URL to latest dump of wikipedia.
	url = "https://download.kiwix.org/zim/wikipedia/"

	# Different mappings of targets to files.
	target_mapping = {
		"all-maxi": "wikipedia_en_all_maxi",
		"all-mini": "wikipedia_en_all_mini",
		"all-nopic": "wikipedia_en_all_nopic",
		"simple-all-maxi": "wikipedia_en_simple_all_maxi",
		"simple-all-mini": "wikipedia_en_simple_all_mini",
		"simple-all-nopic": "wikipedia_en_simple_all_nopic",
	}

	# Verify arguments.
	target = args.target
	valid_targets = list(target_mapping.keys())
	if target not in valid_targets:
		print(f"Download argument 'target' {target} not valid target.")
		print(f"Please specify one of the following for the target: {valid_targets}")
		exit(1)

	# Folder and file path for downloads.
	folder = "./WikipediaData"
	base_file = target_mapping[target]
	if not os.path.exists(folder) or not os.path.isdir(folder):
		os.makedirs(folder, exist_ok=True)
	
	###################################################################
	# QUERY LATEST PAGE TO GET DOWNLOAD URL
	###################################################################
	# Query the download page.
	response = requests.get(url)
	return_status = response.status_code
	if return_status != 200:
		print(f"Request returned {return_status} status code for {url}")
		exit(1)

	# Set up BeautifulSoup object.
	soup = BeautifulSoup(response.text, "lxml")

	# # Find the necessary link.
	links = soup.find_all("a")
	targeted_links = [
		link for link in links 
		if base_file in link.get('href') 
	]
	if links is None or len(targeted_links) == 0:
		print(f"Could not find {base_file} in latest dump {url}")
		exit(1)
	
	# Get link from the main page.
	link_element = targeted_links
	
	# Map each file to the respective url, local filepath, and later
	# SHA1SUM.
	files = {
		link.get("href") : [
			url + link.get("href"), 									# link url
			os.path.join(folder, link.get("href").replace(url, "")),	# local filepath
		] 
		for link in link_element
	}

	###################################################################
	# DOWNLOAD FILE
	###################################################################
	# Initialize the file download.
	print("WARNING!")
	print("The compressed files downloaded can be as large as 100 GB. Please make sure you have enough disk space before proceeding.")
	confirmation = input("Proceed? [Y/n] ")
	if confirmation not in ["Y", "y"]:
		exit(0)

	# Download the latest only.
	latest = sorted(list(files.keys()))[-1]
	print(f"Downloading {latest} file...")
	download_status = downloadFile(
		files[latest][0],	# link url
		files[latest][1], 	# local filepath
	)
	status = "successfully" if download_status else "unsuccessfully"
	status += "."

	msg = f"Target file {latest} was downloaded"
	if args.no_shasum_check:
		msg += "."
	else:
		msg += status
	print(msg)
	
	# Exit the program.
	exit(0)


def downloadFile(url: str, local_filepath: str) -> None:
	"""
	Download the specific (compressed) file from the wikipedia link.
		Verify that the downloaded file is also correct with SHA1SUM.
	@param: url (str), the URL of the compressed file to download.
	@param: local_filepath (str), the local path the compressed file is
		going to be saved to.
	@return: returns whether the file was successfully downloaded and 
		the SHA256SUMs match.
	"""
	# Get the expected SHA256.
	response = requests.get(url + ".sha256")
	if response.status_code != 200:
		return False
	
	sha256 = response.content.decode("utf-8").split()[0]

	# Open request to the URL.
	with requests.get(url, stream=True) as r:
		# Raise an error if there is an HTTP error.
		r.raise_for_status()

		# Open the file and write the data to the file.
		with open(local_filepath, 'wb+') as f:
			for chunk in r.iter_content(chunk_size=8192):
				f.write(chunk)

	# Return whether the file was successfully created.
	new_sha256 = hashSum(local_filepath)
	print(f"Expected SHA256: {sha256}")
	print(f"Computed SHA256: {new_sha256}")
	return sha256 == new_sha256


def hashSum(local_filepath: str) -> str:
	"""
	Compute the SHA1SUM of the downloaded (compressed) file. This
		confirms if the download was successful.
	@param: local_filepath (str), the local path the compressed file is
		going to be saved to.
	@return: returns the SHA256SUM hash.
	"""

	# Initialize the SHA256 hash object.
	sha256 = hashlib.sha256()

	# Check for the file path to exist. Return an empty string if it
	# does not exist.
	if not os.path.exists(local_filepath):
		return ""

	# Open the file.
	with open(local_filepath, 'rb') as f:
		# Read the file contents and update the has object.
		while True:
			data = f.read(8192)
			if not data:
				break
			sha256.update(data)

	# Return the digested hash object (string).
	return sha256.hexdigest()


if __name__ == '__main__':
	main()