class Market:
    def __init__(self, session, base_url):
        self.session = session
        self.base_url = base_url

    def get_quote(self, symbol):
        url = f"{self.base_url}/v1/market/quote/{symbol}.json"
        r = self.session.get(url)
        return r.json() if r.status_code == 200 else None
