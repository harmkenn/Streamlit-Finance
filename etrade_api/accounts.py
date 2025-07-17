class Accounts:
    def __init__(self, session, base_url):
        self.session = session
        self.base_url = base_url

    def get_accounts(self):
        url = f"{self.base_url}/v1/accounts/list.json"
        r = self.session.get(url)
        return r.json() if r.status_code == 200 else None
