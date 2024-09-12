import unittest

from circuits_benchmark.utils.attr_dict import AttrDict
from circuits_benchmark.utils.get_cases import get_cases


class GetCasesTest(unittest.TestCase):
    def test_get_all_cases(self):
        cases = get_cases()
        names = [case.get_name() for case in cases]
        print(names)

        assert len(names) > 35
        assert "ioi" in names
        assert "ioi_next_token" in names
        assert "3" in names
        assert "37" in names

    def test_cases_filtered_by_indices(self):
        args = AttrDict({"indices": "1,2,3"})
        cases = get_cases(args)
        self.assertEqual(len(cases), 3)

    def test_get_cases_works_for_ioi_cases(self):
        args = AttrDict({"indices": "ioi,ioi_next_token"})
        cases = get_cases(args)
        self.assertEqual(len(cases), 2)
