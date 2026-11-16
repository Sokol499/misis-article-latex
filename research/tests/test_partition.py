import random
import unittest
from research.partition import optimal_partition, brute_force_partition, evaluate, synthetic_case

class PartitionTests(unittest.TestCase):
    def test_dp_equals_enumeration_for_heterogeneous_resources(self):
        for n in range(3,10):
            for k in range(2,min(4,n)+1):
                compute,mem,limits,links=synthetic_case(n,k)
                a=optimal_partition(compute,mem,limits,links)
                b=brute_force_partition(compute,mem,limits,links)
                self.assertEqual(a is None,b is None)
                if a:
                    self.assertAlmostEqual(a.predicted_ms,b.predicted_ms,places=8)
                    self.assertEqual(a.boundaries[0],0)
                    self.assertEqual(a.boundaries[-1],n)
    def test_infeasible(self):
        c,m,l,t=synthetic_case(5,3)
        self.assertIsNone(optimal_partition(c,m,[1,1,1],t))
    def test_only_one_possible_partition(self):
        c,m,l,t=synthetic_case(3,3)
        self.assertEqual(optimal_partition(c,m,l,t).boundaries,(0,1,2,3))
    def test_memory_constraints(self):
        c,m,l,t=synthetic_case(12,3)
        result=optimal_partition(c,m,l,t)
        self.assertIsNotNone(result)
        self.assertTrue(all(a<=b for a,b in zip(result.memory_mib,l)))
    def test_bad_dimensions(self):
        with self.assertRaises(ValueError):
            optimal_partition([[1.0]], [1.0,1.0], [10.0])

if __name__ == '__main__': unittest.main()

