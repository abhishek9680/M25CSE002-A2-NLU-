# M25CSE002-A2-NLU-
# M25CSE002-A2-NLU

**Course/Domain:** Natural Language Understanding (NLU) / Computer Science and Engineering  
**Student Name:** Abhishek Gehlot  
**Student ID:** M25CSE002  

This repository contains the complete codebase, datasets, and analytical report for NLU Assignment 2. The project is divided into two comprehensive problems exploring deep learning applications in NLP: training custom word embeddings from scratch and building character-level sequence models for text generation.

---

## 📑 Table of Contents
1. [Project 1: Learning Word Embeddings from Scratch](#project-1-learning-word-embeddings-from-scratch)
2. [Project 2: Character-Level Indian Name Generation](#project-2-character-level-indian-name-generation)
3. [Repository Structure](#repository-structure)
4. [Setup & Update Instructions](#setup--update-instructions)
5. [Execution Guide](#execution-guide)
6. [Key Findings & Report](#key-findings--report)

---

## 🧠 Project 1: Learning Word Embeddings from Scratch

**Objective:** Build an end-to-end NLP pipeline to scrape domain-specific data, construct a vocabulary, and train Word2Vec architectures entirely from scratch using PyTorch (without utilizing high-level libraries like Gensim).

* **Dataset Curation:** Scraped official IIT Jodhpur website pages (departments, courses, admissions, research). The corpus was dynamically augmented with NLTK's `brown` academic corpus to mathematically guarantee a vocabulary size exceeding 6,000 unique tokens.
* **Preprocessing:** Strict pipeline to remove boilerplate HTML, URLs, non-ASCII characters, and digits, followed by lowercasing and stop-word removal.
* **Model Architectures:** Custom PyTorch implementations of **Continuous Bag of Words (CBOW)** and **Skip-gram** models, featuring custom Negative Sampling loss functions and `Adam` optimization.
* **Hyperparameter Grid Search:** Evaluated 72 distinct configurations across Embedding Dimensions (50, 100, 200, 300), Context Windows (3, 5, 7), and Negative Samples (5, 10, 15).
* **Evaluation:** * **Semantic:** Top-5 Nearest Neighbors (Cosine Similarity) and Analogy resolution (e.g., `ug : btech :: pg : ?`).
  * **Visual:** High-dimensional clustering visualization using PCA and t-SNE.

---

## 📝 Project 2: Character-Level Indian Name Generation

**Objective:** Design and compare sequence models to generate novel, diverse, and phonetically realistic Indian names based on a curated dataset of 1,000 regional names.

* **Dataset:** 1,000 diverse Indian names spanning multiple linguistic regions (Hindi, Tamil, Telugu, Bengali, Marathi, etc.).
* **Model Architectures:**
  1. **Vanilla Recurrent Neural Network (RNN)**
  2. **Bidirectional LSTM (BLSTM):** Implemented with a strict `num_layers=1` constraint to prevent backward-pass data leakage during autoregressive generation.
  3. **RNN with Basic Attention:** Utilizes scaled dot-product self-attention with causal masking.
* **Quantitative Evaluation:** Evaluated on **Novelty Rate** (target: 30%-60% to avoid overfitting/underfitting) and **Diversity Rate** (target: ~100%).
* **Qualitative Evaluation:** Names were scored using a custom heuristic `realism_score` function evaluating phonetic rules, vowel/consonant ratios, and prefix/suffix validity.

---

## 📂 Repository Structure

```text
📦 M25CSE002-NLU-Assignment2
 ┣ 📜 NLU P1.ipynb     # Full code pipeline for Word Embeddings (Task 1 to Task 4)
 ┣ 📜 NLU P2.ipynb     # Full code pipeline for Sequence Modeling (Task 0 to Task 3)
 ┣ 📜 Report.pdf                    # Comprehensive analytical report detailing methodologies and results
 ┣ 📜 campus_data.txt               # Raw text corpus data for Problem 1
 ┣ 📂 data/                         # Generated preprocessed text and pickle files
 ┣ 📂 models/                       # Saved PyTorch .pth model weights
 ┗ 📂 outputs/                      # Generated visualizations, plots, and CSV results
