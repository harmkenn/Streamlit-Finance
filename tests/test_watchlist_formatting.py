import importlib.util
import os


MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "watchlist.py"))
spec = importlib.util.spec_from_file_location("watchlist_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_money_formatter_handles_none_values():
    assert module.format_money(None) == "N/A"
    assert module.format_money(1234.5) == "$1,234.50"


def test_pct_formatter_handles_none_values():
    assert module.format_pct(None) == "N/A"
    assert module.format_pct(1.25) == "+1.25%"
