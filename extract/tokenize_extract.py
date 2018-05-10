# coding: utf-8

import re
import math
import codecs
import random
import string
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import normalize


class FileCorpus(object):
    """
    文本语料
    """
    def __init__(self, file_path):
        """
        :param file_path: 文本文件路径
        :type file_path: str
        """
        self.file_path = file_path

    def __iter__(self):
        with codecs.open(self.file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                yield line


def write_file(lines, file_path, mode='w'):
    with codecs.open(file_path, mode, encoding='utf-8') as f:
        f.writelines(lines)


def extract_tokenize(corpus_file, max_word_len=4, min_count=128, min_agg=5, min_entropy=1):
    """
    从语料中提取词
    :param corpus_file: 语料文件
    :param max_word_len:
    :param min_count: 最小词频
    :param min_agg: 最小凝固度，不同ngram不同值
    :param min_entropy: 最小自由度
    :type corpus: str
    :type max_word_len: int
    :type min_count: int
    :type min_agg: float
    :type min_entropy: float
    :return words (w, freqs[w], aggs[w], entropys[w], scores[w])
    :rtype: tuple[str, int, float, float, float]
    """
    # 清洗文本
    corpus = FileCorpus(corpus_file)

    # 子句去重
    corpus = duplicate(corpus)

    ngrams = build_ngrams(corpus, max_word_len, min_count=min_count)

    '''
    凝固度
    '''
    aggs = cal_agg(ngrams, min_agg=min_agg)
    words = [w for w, _ in aggs.items()]

    '''
    词频
    '''
    freqs = defaultdict(int)
    for doc in corpus:
        for w in cut(doc, ngrams, max_word_len):
            freqs[w] += 1
    words = [w for w in words if freqs[w] >= min_count]

    '''
    自由度
    '''
    wls, wrs = count_neighbors(corpus, max_word_len, ngrams, words)

    entropys = {w: min(cal_entropy(wls[w]), cal_entropy(wrs[w])) for w in words}
    words = [w for w in words if entropys[w] >= min_entropy]

    # words = sorted(words, key=lambda w: (freqs[w], aggs[w], entropys[w]), reverse=True)
    # words = sorted(words, key=lambda w: (aggs[w], entropys[w]), reverse=True)

    freq_x = np.array([freqs[w] for w in words])
    agg_x = np.array([aggs[w] for w in words])
    entropy_x = np.array([entropys[w] for w in words])

    freq_n_x = normalize([freq_x])[0]
    agg_n_x = normalize([agg_x])[0]
    entropy_n_x = normalize([entropy_x])[0]

    '''
    将freq, agg, entropy合并成一个度量
    '''
    scores_x = np.log(freq_n_x) + np.log(agg_n_x) + np.log(entropy_n_x)
    scores = {w: score for w, score in zip(words, scores_x)}

    words = sorted(words, key=lambda w: scores[w], reverse=True)

    for w in words:
        if not re.match(r'\d+', w) and not re.match(r'[a-zA-Z]+', w):
            yield (w, freqs[w], aggs[w], entropys[w], scores[w])


def clean_word(text, **kwargs):
    """
    清洗文本
    :param text:
    """
    # 清除html转义字符
    text = re.sub(r'&[a-zA-Z0-9]+;', '', text)
    return [t for t in re.split(r'[^\u4e00-\u9fa50-9a-zA-Z]+', text)]


def build_ngrams(docs, n, min_count=0):
    """
    构建ngrams
    :param docs:
    :param n:
    :param min_count:
    :return:
    """
    ngrams = defaultdict(int)

    for doc in docs:
        for i in range(len(doc)):
            for j in range(1, n + 1):
                if i + j <= len(doc):
                    ngrams[doc[i:i + j]] += 1

    ngrams = {i: j for i, j in ngrams.items() if j >= min_count}
    return ngrams


def cut(txt, dictionary, max_char_num):
    """
    依照词典对文本切词
    :param txt: 待切词的文本
    :param dictionary: 词典
    :param max_char_num: 词的最大字数
    :return:
    """
    if txt is None or txt.strip() == '':
        return []

    txt = txt.strip()

    r = np.array([0] * (len(txt) - 1))
    for i in range(len(txt) - 1):
        for j in range(2, max_char_num + 1):
            if txt[i:i + j] in dictionary:
                r[i:i + j - 1] += 1
    words = [txt[0]]
    for i in range(1, len(txt)):
        if r[i - 1] > 0:
            words[-1] += txt[i]
        else:
            words.append(txt[i])
    return words


def is_real(s, n, ngrams):
    if len(s) >= 3:
        for i in range(3, n + 1):
            for j in range(len(s) - i + 1):
                if s[j:j + i] not in ngrams:
                    return False
        return True
    else:
        return True


def random_string(n=6):
    return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def cal_entropy(neighbors):
    """
    计算邻字熵
    :param neighbors:
    :return:
    """
    e = 0

    total = sum([c for _, c in neighbors.items()])
    for n, c in neighbors.items():
        p = c * 1.0 / total
        e += -p * math.log2(p)

    return e


def cal_agg(ngrams, min_agg=0.0):
    """
    根据凝固度过滤ngram
    ab的凝固度 = p(a, b) / (p(a) * p(b))
    p(x) = #x / #total
    :param ngrams:
    :param min_agg:
    :return:
    """
    scores = defaultdict(float)

    total = 1. * sum([c for s, c in ngrams.items() if len(s) == 1])

    for s, c in ngrams.items():
        if len(s) >= 2:
            score = min([total * ngrams[s] / (ngrams[s[:i + 1]] * ngrams[s[i + 1:]]) for i in range(len(s) - 1)])
            if score >= min_agg:
                scores[s] = score

    return scores


def softmax(x):
    """
    :param x:
    :type x: np.array
    """
    e = np.exp(x)
    return e / np.sum(e)


tmp_file = 'extract.corpus.tmp.txt'


def duplicate(raw_corpus):
    """
    子句去重，去除重复的子句，防止高频重复句子
    :param raw_corpus:
    :return:
    """
    lines = set()
    for line in raw_corpus:
        lines.add('%s\n' % line)
    write_file(lines, tmp_file)
    return FileCorpus(tmp_file)


def count_neighbors(corpus, max_word_len, ngrams, words):
    """
    统计左右邻字
    :param corpus:
    :param max_word_len:
    :param ngrams:
    :param words:
    :return:
    """
    wls = defaultdict(lambda: defaultdict(int))
    wrs = defaultdict(lambda: defaultdict(int))

    pre_ws = []
    curr_ws = []

    for next_line in corpus:

        next_ws = cut(next_line, ngrams, max_word_len)
        if len(next_ws) < 5:
            continue

        for i, w in enumerate(curr_ws):
            if w not in words:
                continue

            if pre_ws and curr_ws:
                wl = pre_ws[-1] if i == 0 else curr_ws[i - 1]
                wls[w][wl] += 1

            if next_ws and curr_ws:
                wr = next_ws[0] if i == len(curr_ws) - 1 else curr_ws[i + 1]
                wrs[w][wr] += 1

        pre_ws = curr_ws
        curr_ws = next_ws

    return wls, wrs

