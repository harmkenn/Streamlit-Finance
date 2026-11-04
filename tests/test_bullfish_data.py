import importlib.util
import os
from unittest.mock import Mock, patch

APP_IMPORT_HTML = """
<table>
  <thead>
    <tr><th>No.</th><th>Symbol</th><th>Company Name</th><th>% Change</th>
    <th>Stock Price</th><th>Volume</th><th>Market Cap</th></tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>ABCD</td><td>Example Company</td><td>12.50%</td>
    <td>1,234.50</td><td>56,789</td><td>1.2M</td></tr>
  </tbody>
</table>
"""

MODULE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "apps", "bullfish.py"))
spec = importlib.util.spec_from_file_location("bullfish_module", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
with patch("requests.get", return_value=Mock(text=APP_IMPORT_HTML)):
    spec.loader.exec_module(module)


def test_parse_stockanalysis_gainers_maps_table_to_dashboard_columns():
    result = module.parse_stockanalysis_gainers(APP_IMPORT_HTML)

    assert result.iloc[0].to_dict() == {
        "Ticker": "ABCD",
        "Name": "Example Company",
        "Price ($)": 1234.5,
        "Change (%)": 12.5,
        "Volume": 56789.0,
        "URL": "https://stockanalysis.com/stocks/abcd/",
    }


def test_parse_stockanalysis_gainers_rejects_missing_table():
    try:
        module.parse_stockanalysis_gainers("<html><body>No table</body></html>")
    except ValueError as exc:
        assert str(exc) == "Gainers table was not found in the response."
    else:
        raise AssertionError("Expected missing-table response to raise ValueError")
