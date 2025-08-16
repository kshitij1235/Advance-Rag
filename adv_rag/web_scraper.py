import requests
from bs4 import BeautifulSoup


class WebScraper:
    @staticmethod
    def clean(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script", "style", "nav", "header", "footer", "aside"]):
            t.decompose()
        txt = soup.get_text(separator=" ", strip=True)
        return "\n".join([l for l in (line.strip() for line in txt.splitlines()) if l])

    def fetch(self, url: str):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return self.clean(r.text)
        except Exception as e:
            print(f"[ERR] fetch {url}: {e}")
            return None
