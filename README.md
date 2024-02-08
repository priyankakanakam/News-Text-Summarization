# News-Text-Summarization

## Overview

This project demonstrates News Text Summarization using a combination of TextRank algorithm and an ensemble technique that averages cosine and BM25 similarity scores. The goal is to generate concise and meaningful summaries from news articles.

## Dependencies

The following Python libraries are used in this project:

- `nltk`: Natural Language Toolkit for text processing.
- `numpy`: Numerical computing library for array operations.
- `networkx`: Library for creating and manipulating graphs.

Ensure that these libraries are installed before running the code.

```bash
pip install nltk numpy networkx seaborn
```

## Usage

Input your url or news text when prompted, and the program will generate a summary using the TextRank algorithm with an ensemble technique of cosine and BM25 similarity scores.

## Algorithm Overview

1. **TextRank Algorithm:**
   - Extracts important sentences from the news text based on sentence similarity in a graph representation.
  
2. **Cosine Similarity:**
   - Calculates similarity scores between sentences using the cosine algorithm.

3. **BM25 Similarity:**
   - Calculates similarity scores between sentences using the BM25 algorithm.

4. **Ensemble Technique:**
   - Combines cosine and BM25 similarity scores through averaging.

5. **Summary Generation:**
   - Ranks sentences using the ensemble similarity matrix and selects the top sentences for the summary.

## Results

The program generates a concise and informative summary of the input news text, providing users with an efficient way to grasp the key information.

Feel free to explore and customize the code according to your needs. Happy summarizing!

**Note:** Ensure that you have Python installed on your system to run the code.
