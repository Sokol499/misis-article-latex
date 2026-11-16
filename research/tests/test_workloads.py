import unittest
from research.workloads import build_workloads

class FakeTokenizer:
    def encode(self,txt,add_special_tokens=False):
        return [ord(c)%251 for c in txt]
    def decode(self,ids,clean_up_tokenization_spaces=False):
        return 'text-'+str(len(ids))

class WorkloadTests(unittest.TestCase):
    def test_count_and_lengths(self):
        xs=build_workloads(FakeTokenizer())
        self.assertEqual(len(xs),144)
        self.assertEqual(len(set(x['case_id'] for x in xs)),144)
        self.assertEqual(set(x['input_tokens'] for x in xs),{16,64,128,256})
        self.assertEqual(set(x['max_new_tokens'] for x in xs),{8,16,32})
        self.assertTrue(all(len(x['input_ids'])==x['input_tokens'] for x in xs))
    def test_overflow(self):
        with self.assertRaises(ValueError):
            build_workloads(FakeTokenizer(),max_position_embeddings=270)

