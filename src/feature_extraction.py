import re
from urllib.parse import urlparse

def extract_features(url):
    if not url.startswith(("http://", "https://")):
     url = "https://" + url
    features = {
        'url_length': len(url),
        'domain_length': len(urlparse(url).netloc),
        'dot_count': url.count('.'),
        'hyphen_count': url.count('-'),
        'digit_count': sum(c.isdigit() for c in url),
        'https': 1 if url.startswith("https") else 0,
        'has_at': 1 if '@' in url else 0,
        'has_ip': 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
    }

    return features