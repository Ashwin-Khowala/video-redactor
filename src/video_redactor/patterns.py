import re
from .config import MatchConfig

EMAIL_REGEX = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
PHONE_REGEX = re.compile(r'(\+?\d[\d\s\-().]{6,20}\d)')

DEFAULT_EMAIL_KEYWORDS = [
    'gmail', 'gmall', 'gmai', 'gma1', 'outlook', 'yahoo', 'hotmail', 'icloud', 
    'protonmail', 'kiit.ac', 'kiitacin', 'clubfyndr', 'digilocker'
]

def is_sensitive(text: str, config: MatchConfig) -> bool:
    text_clean = text.strip()
    if not text_clean:
        return False
        
    if config.mode == 'all' or 'all' in config.redact_types:
        return True
        
    lower_text = text_clean.lower()
    
    if 'keywords' in config.redact_types and config.custom_keywords:
        for kw in config.custom_keywords:
            if kw.lower() in lower_text:
                return True
                
    if 'email' in config.redact_types:
        if "@" in lower_text:
            return True
        for kw in DEFAULT_EMAIL_KEYWORDS:
            if kw in lower_text:
                return True
        if EMAIL_REGEX.search(text_clean):
            return True
            
    if 'phone' in config.redact_types:
        match = PHONE_REGEX.search(text_clean)
        if match:
            matched_phone = match.group(0)
            matched_digits = "".join(c for c in matched_phone if c.isdigit())
            if 7 <= len(matched_digits) <= 15:
                if "-" in matched_phone and len(matched_digits) == 8:
                    return False
                return True
            
    return False
