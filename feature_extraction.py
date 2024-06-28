import re
from urllib.parse import urlparse
# Function to extract features from URL
def extract_features(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    path = parsed_url.path
    
    # Length-based features
    url_length = len(url)
    domain_length = len(domain)
    path_length = len(path)
    num_path_segments = len(path.split('/'))
    num_queries = len(parsed_url.query)
    
    # Count-based features
    num_dots = domain.count('.')
    num_hyphens = domain.count('-')
    num_digits = sum(c.isdigit() for c in domain)
    num_special_chars = len(re.findall(r'[^a-zA-Z0-9]', domain))
    num_subdomains = len(domain.split('.'))
    
    # Binary features
    has_https = 1 if parsed_url.scheme == 'https' else 0
    has_dot_com = 1 if domain.endswith('.com') else 0
    
    # Combine all features into a list
    features = [
        url_length, domain_length, path_length, num_path_segments, num_queries,
        num_dots, num_hyphens, num_digits, num_special_chars, num_subdomains,
        has_https, has_dot_com
    ]
    
    return features
