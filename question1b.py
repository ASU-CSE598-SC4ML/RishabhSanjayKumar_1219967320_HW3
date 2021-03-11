#!/usr/bin/env python3
import itertools
import logging
import unittest
from test.multiprocess_test_case import MultiProcessTestCase, get_random_test_tensor

import crypten
import torch
from crypten.common.tensor_types import is_int_tensor
from crypten.mpc.primitives import BinarySharedTensor

class TestBinary(MultiProcessTestCase):

	def _check(self, encrypted_tensor, reference, msg, dst=None, tolerance=None):
		if tolerance is None:
			tolerance = getattr(self, "default_tolerance", 0.05)
		tensor = encrypted_tensor.get_plain_text(dst=dst)
		if dst is not None and dst != self.rank:
			self.assertIsNone(tensor)
			return

		# Check sizes match
		self.assertTrue(tensor.size() == reference.size(), msg)

		self.assertTrue(is_int_tensor(reference), "reference must be a long")
		test_passed = (tensor == reference).all().item() == 1
		if not test_passed:
			logging.info(msg)
			logging.info("Result %s" % tensor)
			logging.info("Result - Reference = %s" % (tensor - reference))
		self.assertTrue(test_passed, msg=msg)
        
        
	def test_comparators(self):
		"""Test comparators (>, >=, <, <=, ==, !=)"""
		for tensor_type in [lambda x: x, BinarySharedTensor]:
			
			tensor = torch.tensor([10])   #Alice has number 10
			tensor2 = torch.tensor([5])   #Bob has number 5
			
			#tensor = get_random_test_tensor(size=[1], is_float=False)
			#tensor2 = get_random_test_tensor(size=[1], is_float=False)
			
			encrypted_tensor = BinarySharedTensor(tensor)
			encrypted_tensor2 = tensor_type(tensor2)
			
			reference = getattr(tensor, "gt")(tensor2).long()
			encrypted_out = getattr(encrypted_tensor, "gt")(encrypted_tensor2)
			
			print("Reference: = ", reference)
			print("Encrypted out: = ", encrypted_out)
			
			self._check(encrypted_out, reference, "%s comparator failed" % "gt")

# This code only runs when executing the file outside the test harness
if __name__ == "__main__":
    unittest.main()
