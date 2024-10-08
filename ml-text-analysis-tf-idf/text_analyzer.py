import argparse
from pathlib import Path
import glob
import os
import re
import math
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd


def get_top_k(kv_dict: Dict[str, float], k: int = 20) -> List[Tuple[str, float]]:
    """
    Returns the top 'k' key-value pairs from a dictionary, based on their values.

    :param kv_dict: A dictionary of key-value pairs, where the values are scores or counts.
    :param k: The number of key-value pairs with top 'k' values (default k=20).
    :return: A list of the top 'k' key-value pairs from the dictionary, sorted by value.

    Example:
    # >>> kv_dict = {'apple': 5.0, 'banana': 3.0, 'orange': 2.5, 'peach': 1.0}
    # >>> get_top_k(kv_dict, 2)
    [('apple', 5.0), ('banana', 3.0)]
    """
    # Sort the dictionary by value and return the top 'k' key-value pairs

    sorted_pairs = sort_dictionary_by_value(kv_dict)

    top_k = sorted_pairs[:k]
    return top_k


def sort_dictionary_by_value(
        dict_in: Dict[str, float], direction: str = "descending"
) -> List[Tuple[str, float]]:
    """
    Sort a dictionary of key-value pairs by their values.

    :param dict_in: A dictionary of key-value pairs, where the values are scores or counts.
    :param direction: The sorting direction, either 'descending' (default) or 'ascending'.
    :return: A list of the key-value pairs from the dictionary, sorted by value.

    Example:
    # >>> kv_dict = {'apple': 5.0, 'banana': 3.0, 'orange': 2.5, 'peach': 1.0}
    # >>> sort_dictionary_by_value(kv_dict)
    [('peach', 1.0), ('orange', 2.5), ('banana', 3.0), ('apple', 5.0)]
    """
    # Sort the dictionary  dict_in by value
    # Reverse the order if the direction is 'descending'

    sort_dict = sorted(dict_in.items(), key=lambda x: x[1], reverse=(direction == "descending"))

    return sort_dict


def strip_non_ascii(string):
    """Returns the string without non ASCII characters"""
    stripped = (c for c in string if 0 < ord(c) < 127)
    return "".join(stripped)


def clean_text(s):
    """Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana'"""
    s = strip_non_ascii(s)
    s = re.sub("[^a-z A-Z]", "", s)
    s = s.replace(" n ", " ")
    
    # TODO : Remove if this fails the autograder for capitalization
    s = s.lower()

    return s


def clean_corpus(corpus):
    """Run clean_text() on each sonnet in the corpus

    :param corpus:  corpus dict with keys set as filenames and contents as a single string of the respective sonnet.
    :type corpus:   dict

    :return     corpus with text cleaned and tokenized. Still a dictionary with keys being file names, but contents
                now the cleaned, tokenized content.
    """
    for key in corpus.keys():
        # clean each exemplar (i.e., sonnet) in corpus

        # call function provided to clean text of all non-alphabetical characters and tokenize by " " via split()
        corpus[key] = clean_text(corpus[key]).split()

    return corpus


def read_sonnets(fin):
    """
    Passes image through network, returning output of specified layer.

    :param fin: fin can be a directory path containing TXT files to process or to a single file,

    :return: (dict) Contents of sonnets with filename (i.e., sonnet ID) as the keys and cleaned text as the values.
    """

    """ reads and cleans list of text files, which are sonnets in this assignment"""

    if Path(fin).is_file():
        f_sonnets = [fin]
    elif Path(fin).is_dir():
        f_sonnets = glob.glob(fin + os.sep + "*.txt")
    else:
        print("Filepath of sonnet not found!")
        return None

    sonnets = {}
    for f in f_sonnets:
        sonnet_id = Path(f).stem
        data = []
        with open(f, "r") as file:
            data.append(file.readline().replace("\\n", "").replace("\\r", ""))

        sonnets[sonnet_id] = clean_text("".join(data))
    return sonnets


def tf(document: List[str]) -> Dict[str, int]:
    """
    Calculate the term frequency (TF) for each word in a document.

    The term frequency of a word is defined as the number of times it appears in the document.

    :param document: A list of words representing the document.
    :return: A dictionary where the keys are words and the values are their term frequency in the document.

    Example:
    # >>> doc = ['apple', 'banana', 'orange', 'peach', 'apple']
    # >>> tf(doc)
    {'apple': 2, 'banana': 1, 'orange': 1, 'peach': 1}
    """
    # Count the occurrences of each word in the document
    document_tf = {}  # TODO: fix me

    for word in document:
        if word not in document_tf.keys():
            document_tf[word] = 1
        else:
            document_tf[word] += 1

    return document_tf


def idf(corpus: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate the inverted document frequency (IDF) for each word in a corpus.

    The IDF of a word is defined as log(N/df), where N is the total number of documents in the corpus and df
    is the number of documents that contain the word.

    :param corpus: A dictionary of documents, where each document is represented as a list of words.
    :return: A dictionary where the keys are words and the values are their IDF scores.

    Example:
    # >>> corpus = {"doc1": ["apple", "banana", "orange"], "doc2": ["banana", "peach"], "doc3": ["orange", "peach"]}
    # >>> idf(corpus)
    {'apple': 1.0986122886681098, 'banana': 0.4054651081081644, 'orange': 0.4054651081081644, 'peach': 0.6931471805599453}
    """

    # Calculate the IDF for each word
    corpus_idf = {}

    # get total number of documents in corpus
    N = len(corpus.keys())

    # get number of documents that contain each word

    # iterate over each document in corpus
    for doc in corpus.keys():

        # generate a set of unique words in document
        unique_words = set(corpus[doc])

        # iterate over each word in the set
        for word in unique_words:
            if word not in corpus_idf.keys():
                corpus_idf[word] = 1
            else:
                corpus_idf[word] += 1

    corpus_idf = {
        k: math.log(N / v) 
    for k, v in corpus_idf.items()}

    return corpus_idf


def tf_idf(
        corpus_idf: Dict[str, float], sonnet_tf: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate the TF-IDF scores for each word in a sonnet, using a pre-computed IDF dictionary.

    The TF-IDF score of a word is defined as tf(word) * idf(word), where tf(word) is the term frequency of the word
    in the sonnet and idf(word) is the inverse document frequency of the word in the corpus.

    :param corpus_idf: A dictionary where the keys are words and the values are their IDF scores.
    :param sonnet_tf: A dictionary where the keys are words and the values are their TF scores in the sonnet.
    :return: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.

    Example:
    # >>> corpus_idf = {'apple': 1.0986122886681098, 'banana': 0.4054651081081644, 'orange': 0.4054651081081644, 'peach': 0.6931471805599453}
    # >>> sonnet_tf = {'apple': 2, 'banana': 1, 'orange': 0, 'peach': 3}
    # >>> tf_idf(corpus_idf, sonnet_tf)
    {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'orange': 0.0, 'peach': 2.0794415416798357}
    """

    corpus_tf_idf = { 
        word: sonnet_tf.get(word, 0) * corpus_idf.get(word, 0) 
    for word in sonnet_tf.keys()}

    return corpus_tf_idf


def cosine_sim(
        vec1: Dict[str, float], vec2: Dict[str, float]
) -> float:
    """
    Calculate the cosine similarity between two tf-idf vectors.

    :param vec1: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :param vec2: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :return: The cosine of the vectors

    Example:
    # >>> vec1 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'orange': 0.0}
    # >>> vec2 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'peach': 2.0794415416798357}
    # >>> cosine_sim(vec1, vec2)
    # >>> 0.7320230293693564

    """

    words = set(vec1.keys()).union(set(vec2.keys()))
    vec1 = { word: vec1.get(word, 0) for word in words }
    vec2 = { word: vec2.get(word, 0) for word in words }

    dot_product = { word: vec1[word] * vec2[word] for word in words }
    square_vec1 = { word: vec1[word] ** 2 for word in words }
    square_vec2 = { word: vec2[word] ** 2 for word in words }

    similarity = sum(dot_product.values()) / \
                (math.sqrt(
                    sum(square_vec1.values()) * 
                    sum(square_vec2.values()))
                )

    return similarity


def similarity_matrix(
        func: Callable[[Dict[str, float], Dict[str, float]], float],
        corpus: Dict[str, List[str]],
        corpus_idf: Dict[str, float]
) -> np.ndarray:
    matrix = np.zeros((len(corpus), len(corpus)))
    for i, sonnet1 in enumerate(corpus.values()):
        for j, sonnet2 in enumerate(corpus.values()):
            matrix[i, j] = func(tf_idf(corpus_idf, tf(sonnet1)), tf_idf(corpus_idf, tf(sonnet2)))

    return matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Analysis through TFIDF computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="./data/shakespeare_sonnets/1.txt",
        help="Input text file or files.",
    )
    parser.add_argument(
        "-c",
        "--corpus",
        type=str,
        default="./data/shakespeare_sonnets/",
        help="Directory containing document collection (i.e., corpus)",
    )
    parser.add_argument(
        "--tfidf",
        help="Determine the TF IDF of a document w.r.t. a given corpus",
        action="store_true",
    )

    args = parser.parse_args()

    # return dictionary with keys corresponding to file names and values being the respective contents
    corpus = read_sonnets(args.corpus)

    # return corpus (dict) with each sonnet cleaned and tokenized for further processing
    corpus = clean_corpus(corpus)

    # assign 1.txt to variable sonnet to process and find its TF (Note corpus is of type dic, but sonnet1 is just a str)
    filename = args.input.split("/")[-1].split(".")[0]
    sonnet1 = corpus[filename]

    # determine tf of sonnet
    sonnet1_tf = tf(sonnet1)

    # get sorted list and slice out top 20
    sonnet1_top20 = get_top_k(sonnet1_tf)
    print("\nSonnet 1 TF (Top 20):")
    print(sonnet1_top20)

    # TF of entire corpus
    flattened_corpus = [word for sonnet in corpus.values() for word in sonnet]
    corpus_tf = tf(flattened_corpus)
    corpus_top20 = get_top_k(corpus_tf)
    print("Corpus TF (Top 20):")
    print(corpus_top20)

    # IDF of corpus
    corpus_idf = idf(corpus)
    corpus_tf_ordered = get_top_k(corpus_idf)
    # print top 20 to add to report
    print("Corpus IDF (Top 20):")
    print(corpus_tf_ordered)

    # TFIDF of Sonnet1 w.r.t. corpus
    sonnet1_tfidf = tf_idf(corpus_idf, sonnet1_tf)
    sonnet1_tfidf_ordered = get_top_k(sonnet1_tfidf)
    print("Sonnet 1 TFIDF (Top 20):")
    print(sonnet1_tfidf_ordered)

    # Determine confusion matrix using cosine similarity scores for each exemplar.
    confusion_matrix = similarity_matrix(cosine_sim, corpus, corpus_idf)

    # print confusion matrix to add to report
    print("\nConfusion Matrix:")
    print(confusion_matrix)

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.heatmap(confusion_matrix)
    plt.show()