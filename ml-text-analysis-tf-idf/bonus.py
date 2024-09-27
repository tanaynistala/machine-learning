from pathlib import Path
from typing import Dict, List, Tuple
import math

import numpy as np
import pandas as pd

from text_analyzer import *


def euclidean_distance(
        vec1: Dict[str, float], vec2: Dict[str, float]
) -> float:
    """
    Calculate the Euclidean distance between two tf-idf vectors.

    :param vec1: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :param vec2: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :return: The Euclidean distance between the vectors

    Example:
    # >>> vec1 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'orange': 0.0}
    # >>> vec2 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'peach': 2.0794415416798357}
    # >>> euclidean_distance(vec1, vec2)
    # >>> 2.0794415416798357

    """

    words = set(vec1.keys()).union(set(vec2.keys()))
    vec1 = { word: vec1.get(word, 0) for word in words }
    vec2 = { word: vec2.get(word, 0) for word in words }

    sq_diff = { word : (vec1[word] - vec2[word])**2 for word in words }
    distance = math.sqrt(math.sum(sq_diff.values()))

    return distance


def manhattan_distance(
        vec1: Dict[str, float], vec2: Dict[str, float]
) -> float:
    """
    Calculate the Manhattan distance between two tf-idf vectors.

    :param vec1: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :param vec2: A dictionary where the keys are words and the values are their TF-IDF scores in the sonnet.
    :return: The Manhattan distance of the vectors

    Example:
    # >>> vec1 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'orange': 0.0}
    # >>> vec2 = {'apple': 2.1972245773362196, 'banana': 0.4054651081081644, 'peach': 2.0794415416798357}
    # >>> manhattan_distance(vec1, vec2)
    # >>> 2.0794415416798357

    """

    words = set(vec1.keys()).union(set(vec2.keys()))
    vec1 = { word: vec1.get(word, 0) for word in words }
    vec2 = { word: vec2.get(word, 0) for word in words }

    ab_diff = { word : math.abs(vec1[word] - vec2[word]) for word in words }
    distance = math.sqrt(math.sum(ab_diff.values()))

    return distance


def bm25(corpus: Dict[str, List[str]], corpus_idf: Dict[str, float]) -> float:
    """
    Calculate the BM25 scores for each word in a sonnet, using a pre-computed TF-IDF dictionary.

    :param corpus: A dictionary of documents, where each document is represented as a list of words.
    :param corpus_idf: A dictionary where the keys are words and the values are their IDF scores.
    :return: A dictionary where the keys are words and the values are their BM25 scores in the sonnet.
    """

    # Define constants
    q = 1.25
    b = 0.75

    doc_tfs = { filename : tf(contents) for filename, contents in corpus.items() }
    sizes   = { filename : len(contents) for filename, contents in corpus.items() }
    avg_size = sum(sizes.values()) / len(sizes.values())

    doc_freqs = { 
        word : sum([
            1 if word in doc_tfs[doc] else 0
        for doc in corpus.keys()]) 
    for word in corpus_idf.keys() }

    corpus_bm25 = {
        doc : sum([ 
            doc_tfs[doc][word]
            * (q+1) / (sizes[doc] + q * (1 - b + b * sizes[doc] / avg_size))
            * math.log(1 + (len(corpus.keys()) - doc_freqs[word] + 0.5) / (doc_freqs[word] + 0.5))
        for word in contents ])
    for doc, contents in corpus.items() }

    return corpus_bm25

