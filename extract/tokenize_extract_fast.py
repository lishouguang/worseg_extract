# coding: utf-8

"""
https://spaces.ac.cn/archives/3913/
"""
import re
from collections import defaultdict

from mlang.res.corpus import Corpus


def clean_word(text, **kwargs):
    """
    清洗文本
    :param text:
    """
    # 清除html转义字符
    text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
    return [t for t in re.split(r'[^\u4e00-\u9fa50-9a-zA-Z]+', text)]


def count_words(corpus, min_count=10, min_proba=1, gamma=0.5):
    """
    统计词频，获得不可切分的词组
    :param corpus:
    :param min_count: 最小词频
    :param min_proba: 最小概率
    :param gamma: 平滑数
    :return:
    """
    segments = [defaultdict(int), defaultdict(int)]
    for text in corpus:
        for i in range(2):
            for j in range(len(text)-i):
                segments[i][text[j: j+i+1]] += 1
    segments[0] = {i:j+gamma for i,j in segments[0].items()}
    segments[1] = {i:j+gamma for i,j in segments[1].items()}

    nb_words = sum(segments[0].values())**2/sum(segments[1].values())
    strong_segments = {i: nb_words*j/(segments[0][i[0]]*segments[0][i[1]]) for i,j in segments[1].items() if j >= min_count}
    strong_segments = {i: j for i, j in strong_segments.items() if j >= min_proba}
    return strong_segments


def extract_tokenize(corpus, min_count=5, min_alpha=1):
    """
    从语料中提取词
    :param corpus: 语料文本
    :param min_count: 最小词频
    :param min_alpha: 分词阈值，大于1的数，P(a,b)/P(a)P(b)
    :type corpus: Corpus
    :type min_count: int
    :type min_alpha: float
    :return:
    """
    corpus = Corpus(corpus, record_reader=clean_word, flat=True)

    strong_segments = count_words(corpus, min_count, min_alpha)
    print(strong_segments)

    words = defaultdict(int)
    for text in corpus:
        if text:
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in strong_segments:
                    s += text[i+1]
                else:
                    words[s] += 1
                    s = text[i+1]

    return [(i, j) for i, j in words.items() if j >= min_count]
