import unittest
from research.runner import boundaries_for_layers, predicted_payload, MODES

class RunnerPureTests(unittest.TestCase):
    def test_tiny_boundaries(self):
        self.assertEqual(boundaries_for_layers(2,3),[0,1,2,2])
    def test_large_boundaries(self):
        self.assertEqual(boundaries_for_layers(6,3),[0,2,4,6])
    def test_payloads(self):
        args=(5,20,50257)
        self.assertEqual(predicted_payload('D0',*args),58_298_120)
        self.assertEqual(predicted_payload('D1',*args),4_020_560)
        self.assertEqual(predicted_payload('D2',*args),4_020_560)
        self.assertEqual(predicted_payload('D3',*args),160)
    def test_fair_monolithic_mode(self):
        self.assertIn('B1',MODES)
        self.assertEqual(predicted_payload('B1',16,8,50257),0)
    def test_one_token(self):
        self.assertEqual(predicted_payload('D0',1,1,10),40)
        self.assertEqual(predicted_payload('D3',1,1,10),8)

