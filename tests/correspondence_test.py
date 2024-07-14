from circuits_benchmark.utils.get_cases import get_cases


class TestCorrespondence:
  def test_get_corr_works_on_all_cases(self):
    cases = get_cases()
    for case in cases:
      corr = case.get_correspondence()
      assert corr is not None
