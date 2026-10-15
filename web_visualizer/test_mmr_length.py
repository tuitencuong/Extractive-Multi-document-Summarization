"""Check the 70% MMR policy and preserve its selection through finalization."""
import unittest
from unittest.mock import patch

from methods.main_method.Kmeans_CentroidBase_MMR_SentencePosition import Summarizer
from methods.maximal_marginal_relevance.mmr_summarizer import MMR
from utils.sentence import Sentence


class MMRLengthTests(unittest.TestCase):
    def sentences(self, count):
        return [Sentence('document', ['topic', f'word{i}'], f'Sentence {i}') for i in range(count)]

    def test_candidate_counts_in_both_modes(self):
        for mode in ('train', 'test'):
            for count, expected in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3),
                                    (5, 4), (6, 5), (8, 6), (10, 7), (30, 21)]:
                with self.subTest(mode=mode, count=count):
                    sentences = self.sentences(count)
                    # Negative scores must not stop selection before the count target.
                    with patch.object(MMR, 'MMRScore', return_value=-0.5):
                        result = Summarizer().mmr_summarizer(sentences, list(range(count)), mode=mode)
                    self.assertEqual(len(result), expected)
                    self.assertEqual(len(set(result)), expected)
                    if count:
                        self.assertEqual(result[0], sentences[0].getOGwords())

    def test_count_uses_centroid_pool_not_kmeans_count(self):
        sentences = self.sentences(10)
        candidates = [1, 3, 5, 7]
        result = Summarizer().mmr_summarizer(sentences, candidates)
        self.assertEqual(len(result), 3)
        self.assertTrue(set(result) <= {sentences[i].getOGwords() for i in candidates})

    def test_final_summary_does_not_add_rejected_representatives(self):
        sentences = self.sentences(10)
        texts = [s.getOGwords() for s in sentences]
        summarizer = Summarizer(n_clusters=10, len_summary=10)
        with patch.object(summarizer, 'kmean_summarizer', return_value=(texts, sentences)), \
             patch.object(summarizer, 'centroid_summarizer', return_value=list(range(8))):
            result = summarizer.summary(sentences, [], [], [])
        self.assertEqual(len(result), 6)
        self.assertTrue(set(result) <= set(texts[:8]))


if __name__ == '__main__':
    unittest.main()
