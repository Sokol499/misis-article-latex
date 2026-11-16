import unittest
from research.runner import boundaries_for_layers
from research.partition import optimal_partition, synthetic_case

class BoundaryIntegrationTests(unittest.TestCase):
    def test_cost_model_produces_usable_distribution(self):
        prof=synthetic_case(6,3)
        plan=optimal_partition(*prof)
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.boundaries),4)
        self.assertEqual(plan.boundaries[0],0)
        self.assertEqual(plan.boundaries[-1],6)
        self.assertTrue(all(b>a for a,b in zip(plan.boundaries,plan.boundaries[1:])))
    def test_default_boundaries(self):
        self.assertEqual(boundaries_for_layers(6,3),[0,2,4,6])

