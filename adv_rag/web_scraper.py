import ipaddress
import socket
from urllib.parse import urlparse, urljoin

import requests
from selectolax.parser import HTMLParser


class WebScraper:
    def __init__(self, base_url: str, user_id: str = None, password: str = None,
                 max_bytes: int = 1_000_000):
        """
        base_url:   The base Confluence or company site (e.g. 'https://confluence.mycorp.com/')
        user_id:    Optional username for basic auth
        password:   Optional password for basic auth
        max_bytes:  Maximum size of a page to download
        """
        self.base_url = base_url.rstrip("/")
        self.max_bytes = max_bytes
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "OrgScraper/1.0 (+internal)"
        })

        # attach authentication if provided
        if user_id and password:
            self.session.auth = (user_id, password)

    @staticmethod
    def clean(html: str) -> str:
        """Super-fast RAG-friendly cleaner using selectolax."""
        tree = HTMLParser(html)

        # remove crap
        for node in tree.css("script, style, nav, header, footer, aside, form"):
            node.decompose()

        out_lines = []

        # headings → markdown-style
        for node in tree.css("h1, h2, h3, h4, h5, h6"):
            level = int(node.tag[1])
            out_lines.append(f"\n{'#' * level} {node.text(strip=True)}\n")

        # lists
        for li in tree.css("li"):
            out_lines.append(f"- {li.text(strip=True)}")

        # tables → simple pipe format
        for tr in tree.css("tr"):
            cells = [c.text(" ", strip=True) for c in tr.css("td, th")]
            if cells:
                out_lines.append("| " + " | ".join(cells) + " |")

        # plain text paragraphs
        for p in tree.css("p"):
            txt = p.text(" ", strip=True)
            if txt:
                out_lines.append(txt)

        # normalize + join
        return "\n".join(line.strip() for line in out_lines if line.strip())

    def _is_public_ip(self, host: str) -> bool:
        """
        Resolve host and make sure it’s not localhost/private.
        For org-only mode we ALLOW private IPs (10.x, 192.168.x, etc.)
        since Confluence/Jira/etc. often lives there.
        """
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return False
        for fam, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            try:
                ip_obj = ipaddress.ip_address(ip)
            except ValueError:
                return False
            # allow both global and private IPs for internal projects
            if ip_obj.is_loopback or ip_obj.is_unspecified:
                return False
        return True

    def fetch(self, url: str):
        """
        Fetch a page and return cleaned text, or None on failure.
        - url can be full URL or relative (joined with base_url)
        """
        try:
            # normalize against base URL
            if not url.lower().startswith("http"):
                url = urljoin(self.base_url + "/", url.lstrip("/"))

            parsed = urlparse(url)
            host = parsed.hostname
            if not host:
                raise ValueError("No hostname")

            if not self._is_public_ip(host):
                raise ValueError("Bad host (loopback/invalid)")

            # GET with timeout + byte cap
            with self.session.get(url, timeout=(5, 10), stream=True) as r:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "").lower()
                if "html" not in ctype:
                    raise ValueError(f"Unsupported content type: {ctype}")

                # read safely with cap
                content = b""
                for chunk in r.iter_content(8192):
                    content += chunk
                    if len(content) > self.max_bytes:
                        raise ValueError("Content too large")
                html = content.decode(r.encoding or "utf-8", errors="replace")

            return self.clean(html)

        except Exception as e:
            print(f"[ERR] fetch {url}: {e}")
            return None
