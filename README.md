# M25CSE002-A2-NLU-
# M25CSE002-A2-NLU

**Course:** Natural Language Understanding (NLU) 
**Student Name:** Abhishek Gehlot  
**Student ID:** M25CSE002  

This repository contains two Jupyter Notebooks (`NLU P1.ipynb` and `NLU P2.ipynb`) focused on core Natural Language Understanding tasks: Corpus creation, Exploratory Data Analysis, Word Embeddings, and Character-level Language Modeling (Name Generation).

---

## 📑 Table of Contents
1. [Project 1: Learning Word Embeddings from Scratch](#problem-1-learning-word-embeddings-from-scratch)
2. [Project 2: Character-Level Indian Name Generation](#problem-2-character-level-indian-name-generation)
3. [Repository Contents](#repository-contents)
4. [Repository Structure](#repository-structure)
5. [Prerequisites and Setup](#Prerequisites and Setup)


---

##  Problem 1: Learning Word Embeddings from IITJ Jodhpur

**Objective:** Build an end-to-end NLP pipeline to scrape domain-specific data, construct a vocabulary, and train Word2Vec architectures entirely from scratch using PyTorch (without utilizing high-level libraries like Gensim).

* **Dataset Curation:** Scraped official IIT Jodhpur website pages (departments, courses, admissions, research). The corpus was dynamically augmented with NLTK's `brown` academic corpus to mathematically guarantee a vocabulary size exceeding 6,000 unique tokens.
* **Preprocessing:** Strict pipeline to remove boilerplate HTML, URLs, non-ASCII characters, and digits, followed by lowercasing and stop-word removal.
* **Model Architectures:** Custom PyTorch implementations of **Continuous Bag of Words (CBOW)** and **Skip-gram** models, featuring custom Negative Sampling loss functions and `Adam` optimization.
* **Hyperparameter Grid Search:** Evaluated 72 distinct configurations across Embedding Dimensions (50, 100, 200, 300), Context Windows (3, 5, 7), and Negative Samples (5, 10, 15).
* **Evaluation:** * **Semantic:** Top-5 Nearest Neighbors (Cosine Similarity) and Analogy resolution (e.g., `ug : btech :: pg : ?`).
  * **Visual:** High-dimensional clustering visualization using PCA and t-SNE.

---

## 📝 Problem 2: Character-Level Indian Name Generation

**Objective:** Design and compare sequence models to generate novel, diverse, and phonetically realistic Indian names based on a curated dataset of 1,000 regional names.

* **Dataset:** 1,000 diverse Indian names spanning multiple linguistic regions (Hindi, Tamil, Telugu, Bengali, Marathi, etc.).
* **Model Architectures:**
  1. **Vanilla Recurrent Neural Network (RNN)**
  2. **Bidirectional LSTM (BLSTM):** Implemented with a strict `num_layers=1` constraint to prevent backward-pass data leakage during autoregressive generation.
  3. **RNN with Basic Attention:** Utilizes scaled dot-product self-attention with causal masking.
* **Quantitative Evaluation:** Evaluated on **Novelty Rate** (target: 30%-60% to avoid overfitting/underfitting) and **Diversity Rate** (target: ~100%).
* **Qualitative Evaluation:** Names were scored using a custom heuristic `realism_score` function evaluating phonetic rules, vowel/consonant ratios, and prefix/suffix validity.

---


## 📁 Repository Contents

### 1. `NLU P1.ipynb` (Dataset Preparation & Word Embeddings)
This notebook is dedicated to building a text corpus from scratch, preprocessing it, and exploring text semantics using word embeddings.
* **Task 1: Dataset Preparation (Comprehensive Corpus):** * Attempts to load local data (`campus_data.txt` i.e corpus.txt). 
  * If unavailable, it implements a web scraper using `BeautifulSoup` to crawl the IIT Jodhpur website (CSE courses, academic regulations, faculty pages, etc.) to build a comprehensive corpus.
  * Cleans and tokenizes the text using `NLTK` (removing stopwords, punctuation, etc.) and saves the output to `data/clean_corpus.txt`.
* **Exploratory Data Analysis (EDA):** Generates dataset statistics, computes the top 20 most frequent words, plots frequency bar charts, and generates a Word Cloud.
* **Task 2: Word Embeddings (CBOW & Skip-gram):** Trains/evaluates Word2Vec models (Continuous Bag of Words and Skip-gram) to perform semantic analogy tasks (e.g., evaluating cosine similarity for analogies like `teacher : class :: student : ?`).
*  **Task 3: Report top 5 nearest neighbour for words like research , student , phd, exam:** *
*   **Task 4: Use PCA or t-SNE to project selected word embeddings into 2D space. Visualize clusters for:
       Provide interpretation of clustering behavior and differences between CBOW and Skip-gram.:** *

### 2. `NLU P2.ipynb` (Indian Name Generation)
This notebook focuses on Character-Level Recurrent Neural Networks (RNNs) for text generation. 
* **Task 0: Dataset Generation:** Extracts and parses raw tokens to build a dataset of 1,000 unique Indian names, saving them to `TrainingNames.txt`. Generates dataset statistics (average length, common initials/endings).
* **Task 1: Architecture Profiles:** Trains three different neural network architectures for character-level language modeling:
  * Vanilla RNN
  * Bidirectional LSTM (BiLSTM)
  * RNN + Attention Network
* **Task 2: Quantitative Benchmarks:** Evaluates the generative models based on **Novelty** (generating names not in the training set) and **Diversity** metrics.
* **Task 3: Qualitative analysis:** Discuss: Realism of generated names and Common failure modes. Provide representative
generated samples for each model.
* **Artifact Generation:** Produces comparative loss plots (`all_loss.png`) and qualitative/quantitative evaluation metrics.

---

## 📂 Repository Structure

```text
📦 M25CSE002-NLU-Assignment2
 ┣ 📜 NLU P1.ipynb     # Full code pipeline for Word Embeddings (Task 1 to Task 4)
 ┣ 📜 NLU P2.ipynb     # Full code pipeline for Sequence Modeling (Task 0 to Task 3)
 ┣ 📜 Report.pdf                    # Comprehensive analytical report detailing methodologies and results
 ┣ 📜 campus_data.txt               # Raw text corpus data for Problem 1

---


## 📁 Prerequisites and Setup 

To run these notebooks, you need Python 3.x and the following libraries installed. 

You can install the required dependencies using `pip`:

```bash
pip install gensim nltk beautifulsoup4 requests matplotlib scikit-learn wordcloud pandas tabulate numpy lxml
