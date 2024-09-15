import pytest

from circuits_benchmark.benchmark.cases.case_3 import Case3
from circuits_benchmark.benchmark.cases.case_9 import Case9


class TestTracrDataset:

    @pytest.mark.parametrize("case", [Case3(), Case9()])
    def test_get_encoded_dataset(self, case):
        data = case.get_clean_data()
        assert len(data) == 10
