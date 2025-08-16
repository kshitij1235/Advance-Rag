from adv_rag.web_scraper import WebScraper

def test_clean_html():
    html = "<html><body><h1>Title</h1><script>bad()</script></body></html>"
    text = WebScraper.clean(html)
    assert "Title" in text
    assert "bad()" not in text
