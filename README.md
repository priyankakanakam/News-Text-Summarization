# News-Text-Summarization

## Overview

This project demonstrates News Text Summarization using a combination of TextRank algorithm and an ensemble technique that averages cosine and BM25 similarity scores. The goal is to generate concise and meaningful summaries from news articles.

## Dependencies

The following Python libraries are used in this project:

- `nltk`: Natural Language Toolkit for text processing.
- `rank_bm25`: Implementation of BM25 algorithm for information retrieval.
- `numpy`: Numerical computing library for array operations.
- `networkx`: Library for creating and manipulating graphs.
- `seaborn`: Data visualization library based on Matplotlib.

Ensure that these libraries are installed before running the code.

```bash
pip install nltk rank_bm25 numpy networkx seaborn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/News-Summarization.git
cd News-Summarization
```

2. Run the `summarize.py` script:

```bash
python summarize.py
```

3. Input your news text when prompted, and the program will generate a summary using the TextRank algorithm with an ensemble technique of cosine and BM25 similarity scores.

## Code Structure

- `summarize.py`: Main script for news text summarization.
- `text_rank.py`: Module containing TextRank algorithm implementation.
- `bm25_similarity.py`: Module containing BM25 similarity calculation.
- `ensemble_similarity.py`: Module for combining cosine and BM25 similarity scores.
- `utils.py`: Utility functions for text preprocessing and graph creation.

## Algorithm Overview

1. **TextRank Algorithm:**
   - Extracts important sentences from the news text based on sentence similarity in a graph representation.

2. **BM25 Similarity:**
   - Calculates similarity scores between sentences using the BM25 algorithm.

3. **Ensemble Technique:**
   - Combines cosine and BM25 similarity scores through averaging.

4. **Summary Generation:**
   - Ranks sentences using the ensemble similarity matrix and selects the top sentences for the summary.

## Results

The program generates a concise and informative summary of the input news text, providing users with an efficient way to grasp the key information.

Feel free to explore and customize the code according to your needs. Happy summarizing!

**Note:** Ensure that you have Python installed on your system to run the code.
