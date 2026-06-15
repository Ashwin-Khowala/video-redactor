import pytest
from video_redactor.patterns import is_sensitive
from video_redactor.config import MatchConfig

@pytest.fixture
def cfg():
    return MatchConfig(mode='patterns')

def test_email_at_symbol(cfg):
    assert is_sensitive("user@gmail.com", cfg) is True

def test_email_keyword(cfg):
    assert is_sensitive("gmail", cfg) is False
    assert is_sensitive("outlook", cfg) is False
    assert is_sensitive("kiit.ac", cfg) is False
    assert is_sensitive("randomword", cfg) is False
    assert is_sensitive(".com", cfg) is False
    assert is_sensitive("@", cfg) is False
    assert is_sensitive("testusergmail", cfg) is False


def test_phone_indian(cfg):
    assert is_sensitive("+91 9876543210", cfg) is True

def test_phone_us(cfg):
    assert is_sensitive("+1 (555) 555-5555", cfg) is True

def test_date_not_flagged(cfg):
    assert is_sensitive("2026-06-10", cfg) is False

def test_version_not_flagged(cfg):
    assert is_sensitive("v2.0.1", cfg) is False

def test_mode_all(cfg):
    cfg.mode = 'all'
    assert is_sensitive("hello world", cfg) is True

def test_custom_keyword(cfg):
    cfg.custom_keywords = ["john"]
    assert is_sensitive("john doe", cfg) is True

def test_custom_keyword_standalone(cfg):
    cfg.custom_keywords = ["john"]
    assert is_sensitive("john doe", cfg) is True
    assert is_sensitive("johnny doe", cfg) is False

def test_email_fuzzy_matching(cfg):
    assert is_sensitive("testuser @ gmail . com", cfg) is True
    assert is_sensitive("testuser@gmail com", cfg) is True
    assert is_sensitive("testuser.work@gmail.com", cfg) is True
    assert is_sensitive("12345678@kiit.ac in", cfg) is True
    assert is_sensitive("testuser@gmailcom", cfg) is True
    assert is_sensitive("testuserO6@gmailcom", cfg) is True
    assert is_sensitive("testusero6@gmail:", cfg) is True
    assert is_sensitive("testusergmailcom", cfg) is True
    assert is_sensitive("testusergmail", cfg) is False
    assert is_sensitive("to continue to Club Fyndr", cfg) is False
    assert is_sensitive("Sign in with Gmail", cfg) is False

def test_selective_redaction(cfg):
    cfg.redact_types = ["email"]
    assert is_sensitive("user@gmail.com", cfg) is True
    assert is_sensitive("+91 9876543210", cfg) is False

    cfg.redact_types = ["phone"]
    assert is_sensitive("user@gmail.com", cfg) is False
    assert is_sensitive("+91 9876543210", cfg) is True
