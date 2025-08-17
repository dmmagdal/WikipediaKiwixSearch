# Simple Wikipedia Search

Description: Similar to the WikipediaEnSearch (which operates on the full Wikipedia dumps), this repository focuses on applying the same search techniques on the Simple Wikipedia dumps.


### Setup

 - Environment
 - Download data
 - Preprocess data


### Minimum Hardware Specifications

 - 8 GB RAM if running download script `download.py`.
 - 8 GB RAM if running preprocessing script `preprocess.py` with minimal workers.
     - Note: Was able to run with 8 workers on 8GB RAM and still work just fine.
 - 24 GB RAM if running the precompute sparse vectors script `precompute_sparse_vectors.py` with one worker.
     - Admittedly, this may be a more RAM intensive than realized.



### Notes

 - Kiwix
     - I decided to use Kiwix because they already do a lot of the work of cleaning up the data for mobile use.
     - They have many different copies of wikipedia, including:
         - Wikipedia in different languages.
         - Full wikipedia.
         - Simple wikipedia.
         - Along with curated repositories of wikipedia for different topics/subjects.
     - I pulled my wikipedia dumps from [here](https://download.kiwix.org/zim/wikipedia/).
         - Other libraries from Kiwix can be found [here](https://download.kiwix.org/zim/).
         - To validate the `.zim` files, simply append `.sha256` to the end of the URL used to download the `.zim` file. This will lead to the file with the associated SHA256SUM for that file.
     - I had to use `libzim` in python because Kiwix uses `.zim` files.
         - Offers better memory overhead due to lazy loading.
 - 
 - I should go back an rework the preprocessing pipeline to use rust or something. The time it is taking to run through the dataset (despite it being around 1.8 GB uncompressed vs 95 GB in the uncompressed full Wikipedia dataset) is not acceptable.
     - Possible optimizations:
         - Rewrite in rust.
         - Cut down/remove/ignore all "redirect" pages.
     - In a similar vein, I may have to rework the `preprocess_sparse_vectors.py` script to also be more memory efficient (speed is actually manageable). See above for RAM usage under a single worker for this script.
 - Running the vector database on the entire Simple Wikipedia corpus is still not scalable. Was barely 1/2 way through the first decompressed `.xml` file shard and it had generated over 312 GB of embeddings. A previous napkin calculation for the original full Wikipedia corpus showed that it would need terabytes of storage, so it stands to reason that a similar need would be required for this dataset. 
 - Had to copy the key attributes in `corpus_stats.json` into `config.json`. May want to adjust `precompute_sparse_vectors.py` to automatically update the `config.json` directly.
 - Toggle `corpus_size` affects TF-IDF directly when doing `compute_idf()`.
     - Specifically, for words that are OOV, the smoothed version of the IDF is computed (smoothed version comes from BM25).
 - Really there is a problem with isolating "good" segments of articles from the corpus for testing.
     - Current "bad" quality passages are primarily tables or references/citations.
     - Have to deal with the weird format that WikiMedia uses to insert graph/table data. Hard to parse with normal/simple means.
     - Tried doing tricky weighting schemes to favor longer articles and longer passages. Gets better results but still not good in terms of quality.
     - Applied log to the lengths of the articles so as to balance the weights for sampling the articles. This has also improved results a bit too. 