import importlib.util
import os
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "shortsell.py"))
spec = importlib.util.spec_from_file_location("shortsell_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
resolve_market_source_url = module.resolve_market_source_url


class MarketSourceUrlTests(unittest.TestCase):
    def test_returns_premarket_before_market_open(self):
        now = datetime(2026, 8, 24, 9, 29, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(resolve_market_source_url(now), "https://stockanalysis.com/markets/premarket/")

    def test_returns_gainers_after_market_open(self):
        now = datetime(2026, 8, 24, 9, 31, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(resolve_market_source_url(now), "https://stockanalysis.com/markets/gainers/")

    def test_returns_gainers_on_weekend(self):
        now = datetime(2026, 8, 23, 8, 0, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertEqual(resolve_market_source_url(now), "https://stockanalysis.com/markets/gainers/")


if __name__ == "__main__":
    unittest.main()
