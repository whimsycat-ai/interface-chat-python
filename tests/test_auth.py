"""Tests for auth module."""

import time
import pytest
from claude_interface import is_expired, is_oauth_token, OAuthCredentials


class TestIsExpired:
    def test_returns_true_when_expired(self):
        credentials = OAuthCredentials(
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=int(time.time() * 1000) - 1000,  # 1 second ago
        )
        assert is_expired(credentials) is True

    def test_returns_false_when_not_expired(self):
        credentials = OAuthCredentials(
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=int(time.time() * 1000) + 60000,  # 1 minute from now
        )
        assert is_expired(credentials) is False

    def test_returns_true_when_expires_now(self):
        credentials = OAuthCredentials(
            access_token="sk-ant-oat-test",
            refresh_token="refresh-test",
            expires_at=int(time.time() * 1000),
        )
        assert is_expired(credentials) is True


class TestIsOAuthToken:
    def test_returns_true_for_oauth_tokens(self):
        assert is_oauth_token("sk-ant-oat-abcd1234") is True
        assert is_oauth_token("sk-ant-oat-xyz789") is True

    def test_returns_false_for_api_keys(self):
        assert is_oauth_token("sk-ant-api01-abcd1234") is False
        assert is_oauth_token("sk-test-key") is False
        assert is_oauth_token("api-key-123") is False

    def test_returns_false_for_empty_strings(self):
        assert is_oauth_token("") is False
