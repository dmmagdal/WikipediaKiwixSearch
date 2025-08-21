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
     - Used 16 - 43 GB RAM (and took around 10.5 hours) when running with 28 workers.
         - I think the actual usage was around 27 GB RAM.
 - 20 GB RAM if running the precompute sparse vectors script `precompute_sparse_vectors.py` with one worker.
     - Admittedly, this may be a more RAM intensive than realized but is still far less than the `preprocess.py` script.
     - Depending on CPU, can take as long as 20 minutes to an hour to finish with a single worker.



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
 - Improvements (TODOs)
     - I should go back an rework the preprocessing pipeline to use rust or something. The time it is taking to run through the dataset is rough even on higher end consumer hardware (see above notes for `preprocess.py` script, which was run on a higher end desktop).
         - Possible optimizations:
             - Rewrite in rust.
         - In a similar vein, I may have to rework the `preprocess_sparse_vectors.py` script to also be more memory efficient (speed is actually manageable). See above for RAM usage under a single worker for this script.
     - Running the vector database on the entire Simple Wikipedia corpus is still not scalable. A previous napkin calculation for the original full Wikipedia corpus showed that it would need terabytes of storage, so it stands to reason that a similar need would be required for this dataset. 
         - One possible idea is to only encode passages that are longer than a fixed length (ie word count or number of tokens). Would require some experimentation.
         - Another possible idea is to have a different chunking algorithm that tries to maximize the number of paragraphs together per embedding.
             - This will handle paragraphs that are only a few words (i.e. titles or sub sections) and merge them with longer passages.
     - Had to copy the key attributes in `corpus_stats.json` into `config.json`. May want to adjust `precompute_sparse_vectors.py` to automatically update the `config.json` directly.
         - Toggle `corpus_size` affects TF-IDF directly when doing `compute_idf()`.
     - For words that are OOV, the smoothed version of the IDF is computed (smoothed version comes from BM25).
     - Need to rename some variables to align with proper description of items. Specifically, what was once the SHA1 for the articles (in the xml files) is now the entry ID (in the zim files). This has been updated in some areas but needs to be changed in others. Chunk ID is a number (int) and needs to be treated as such in specific circumstances, while also needing to be treated as a string in others.
 - Automated Testing
     - Really there is a problem with isolating "good" segments of articles from the corpus for testing.
         - Current "bad" quality passages are primarily tables or references/citations.
         - Tried doing tricky weighting schemes to favor longer articles and longer passages. Gets better results but still not good in terms of quality.
         - Applied log to the lengths of the articles so as to balance the weights for sampling the articles. This has also improved results a bit too. 
     - Should look at more models to leverage for generating synthetic queries from randomly sampled passages.
         - So far, I have tried TinyLlama 1.1B, Llama 3.2 1B, BART-large, and Flan-T5. Only Flan-T5 seemed to have provided relatively targeted and on-topic questions compared to the others.
         - Out of the Flan-T5 family, I've only tested Flan-T5 large.
         - Models still need to be able to run on consumer hardware on full FP32 precision. Check back to the hardware spec notes above.
             - Unfortunately, I'm getting the feeling that min specs will be around the 32GB RAM section.
             - However, since the models are best run on the GPU and 32GB VRAM is a bit hard to come by, something like 12 GB VRAM would probably be the reasonable.
             - If GPU is not viable, then fall back on that 32GB RAM min-spec requirement.
     - I've tried what I can in terms of keeping things reproduceable. This could still be worked on for the testing.