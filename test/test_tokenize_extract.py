# coding: utf-8

import unittest

from extract.tokenize_extract import extract_tokenize
from extract.tokenize_extract_fast import extract_tokenize as extract_tokenize_fast


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

    def test_extract_tokenize_fast(self):
        self.assertTrue(True)

        words = extract_tokenize_fast('../data/mobile.sample.txt', min_alpha=2)

        words = sorted(words, key=lambda d: d[1], reverse=True)

        for w, c in words:
            print(w, c)

    def test_extract_tokenize(self):
        self.assertTrue(True)

        words = extract_tokenize('../data/mobile.sample.txt')
        words = [_ for _ in words]

        print('word \t freq \t agg \t entropy \t score')
        for w in words:
            print(w)
            # (w, freqs[w], aggs[w], entropys[w], scores[w])


if __name__ == '__main__':
    unittest.main()
