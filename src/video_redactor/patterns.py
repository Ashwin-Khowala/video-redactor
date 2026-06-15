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
    
    # Custom keywords: standalone check using word boundaries
    if 'keywords' in config.redact_types and config.custom_keywords:
        for kw in config.custom_keywords:
            pattern = re.compile(rf'\b{re.escape(kw.lower())}\b')
            if pattern.search(lower_text):
                return True
                
    # Email detection: simple and aggressive checks
    if 'email' in config.redact_types:
        text_no_spaces = lower_text.replace(" ", "").replace("\t", "").replace("\n", "")
        
        # 1. Check if it contains '@' and has a username and domain around it
        if re.search(r'[a-z0-9_.+-]+@[a-z0-9.-]+', text_no_spaces):
            return True
            
        # 2. Check for default email keywords with a username prefix (at least 3 characters) and a TLD suffix
        kw_pattern = '|'.join(re.escape(kw) for kw in DEFAULT_EMAIL_KEYWORDS)
        tld_pattern = r'com|in|co|org|net|edu|ac|acin'
        if re.search(rf'[a-z0-9_.+-]{{3,}}(?:{kw_pattern})(?:\.|dot)?(?:{tld_pattern})', text_no_spaces):
            return True
            
        # 3. Fallback to standard email regex on raw text
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
