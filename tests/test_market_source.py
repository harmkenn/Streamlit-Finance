import importlib.util
import os
import unittest

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "shortsell.py"))
spec = importlib.util.spec_from_file_location("shortsell_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
get_configured_tickers = module.get_configured_tickers


class TickerListTests(unittest.TestCase):
    def test_reads_tickers_from_repo_file(self):
        self.assertEqual(get_configured_tickers(), ["TQQQ", "UPRO", "^VIX"])


if __name__ == "__main__":
    unittest.main()
