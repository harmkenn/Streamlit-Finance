import importlib.util
import os

import pandas as pd


MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "fishing.py"))
spec = importlib.util.spec_from_file_location("fishing_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_highlight_high_gainers_uses_yellow_for_100_and_green_for_150():
    yellow_row = pd.Series({"Price": 10, "Change %": 149.9})
    green_row = pd.Series({"Price": 10, "Change %": 150.1})

    yellow_style = module.highlight_high_gainers(yellow_row, threshold=150.0)
    green_style = module.highlight_high_gainers(green_row, threshold=150.0)

    assert yellow_style == ["background-color: #f4d03f; color: #000000; font-weight: bold;"] * len(yellow_row)
    assert green_style == ["background-color: #1b5e20; color: #ffffff; font-weight: bold;"] * len(green_row)
