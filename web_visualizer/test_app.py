"""Regression checks for the visualizer's length contract and real cluster traces."""
import unittest
import math
from unittest.mock import patch

from web_visualizer.app import app


class VisualizerTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.tokenizers = patch.multiple('web_visualizer.app', sent_tokenize=None, word_tokenize=None)
        self.tokenizers.start()
        self.addCleanup(self.tokenizers.stop)

    def post(self, count, length=None):
        text = ' '.join(
            f'Tài liệu nghiên cứu chủ đề lĩnh vực nhóm {i % 7} kết quả khảo sát mẫu dữ liệu mã{i}.'
            for i in range(count)
        )
        form = {'texts': text}
        if length is not None:
            form['summaryLength'] = length
        response = self.client.post('/api/summarize', data=form)
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def test_auto_lengths_and_real_members_at_boundaries(self):
        for count, expected in [(1, 1), (19, 3), (20, 5), (49, 5), (50, 8), (99, 8), (100, 10)]:
            with self.subTest(count=count):
                data = self.post(count, '')
                self.assertEqual(data['stats']['summaryMode'], 'auto')
                self.assertEqual(data['stats']['clusterCount'], expected)
                target = math.ceil(0.7 * data['stats']['centroidCount'])
                self.assertEqual(data['stats']['mmrCount'], target)
                self.assertEqual(data['stats']['summaryLength'], target)
                self.assertTrue(all(s['reason'] == 'mmr' for s in data['summary']))
                members = [sid for cluster in data['steps']['clusters'] for sid in cluster['members']]
                self.assertEqual(len(members), count)
                self.assertEqual(len(set(members)), count)
                for cluster in data['steps']['clusters']:
                    self.assertEqual(cluster['size'], len(cluster['members']))
                selected = {s['id'] for s in data['summary']}
                highlighted = {s['id'] for s in data['steps']['preprocess']['sentences'] if s['selected']}
                self.assertEqual(selected, highlighted)

    def test_missing_length_is_auto(self):
        self.assertEqual(self.post(5)['stats']['summaryMode'], 'auto')

    def test_manual_length_limits_mmr_output(self):
        data = self.post(100, '1')
        self.assertGreater(data['stats']['mmrCount'], 1)
        self.assertEqual(len(data['summary']), 1)
        self.assertEqual(data['summary'][0]['id'], data['steps']['preprocess']['sentences'][data['steps']['mmr'][0]['sentenceIndex']]['id'])
        data = self.post(4, '16')
        self.assertEqual(len(data['summary']), data['stats']['mmrCount'])
        self.assertEqual(data['stats']['mmrCount'], math.ceil(0.7 * data['stats']['centroidCount']))

    def test_invalid_lengths_and_empty_input(self):
        for length in ['0', '-1', '17', '1.5', 'abc']:
            with self.subTest(length=length):
                self.assertEqual(self.client.post('/api/summarize', data={'summaryLength': length}).status_code, 400)
        self.assertEqual(self.client.post('/api/summarize', data={}).status_code, 400)
        self.assertEqual(self.client.get('/').status_code, 200)

    def test_lambda_controls_real_mmr_scores(self):
        from methods.maximal_marginal_relevance.mmr_summarizer import MMR
        original_score = MMR.MMRScore
        text = ' '.join(f'Tài liệu nghiên cứu lĩnh vực nhóm {i % 7} khảo sát dữ liệu mã{i}.' for i in range(100))
        with patch.object(MMR, 'MMRScore', autospec=True, side_effect=original_score) as score:
            response = self.client.post('/api/summarize', data={'texts': text, 'lambda': '0.2'})
        self.assertEqual(response.status_code, 200, response.get_json())
        data = response.get_json()
        self.assertEqual(data['stats']['lambda'], 0.2)
        self.assertTrue(score.call_args_list)
        self.assertTrue(all(call.args[4] == 0.2 for call in score.call_args_list))
        for step in data['steps']['mmr'][1:]:
            self.assertAlmostEqual(step['mmrScore'], 0.2 * step['relevance'] - 0.8 * step['redundancy'])
        for invalid in ['-0.1', '1.1', 'nan', 'inf', 'abc']:
            self.assertEqual(self.client.post('/api/summarize', data={'texts': text, 'lambda': invalid}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
