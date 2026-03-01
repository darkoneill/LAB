"""
Tests for openclaw/security/secrets.py and Settings auto-encrypt/decrypt.
"""

import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from openclaw.security.secrets import (
    _default_key_path,
    decrypt_value,
    encrypt_value,
    is_encrypted,
    reset_fernet_cache,
)


@pytest.fixture(autouse=True)
def _isolated_master_key(tmp_path):
    """Point master key at a temp directory and reset the Fernet cache."""
    key_path = tmp_path / "master.key"
    with patch.dict(os.environ, {"OPENCLAW_MASTER_KEY_PATH": str(key_path)}):
        reset_fernet_cache()
        yield key_path
    reset_fernet_cache()


# ── Core encrypt / decrypt ──────────────────────────────────


class TestEncryptDecryptRoundtrip:
    def test_roundtrip_basic(self):
        cipher = encrypt_value("sk-ant-secret-42")
        assert cipher.startswith("ENC:")
        assert decrypt_value(cipher) == "sk-ant-secret-42"

    def test_roundtrip_unicode(self):
        cipher = encrypt_value("clé-secrète-éàü")
        assert decrypt_value(cipher) == "clé-secrète-éàü"

    def test_roundtrip_empty_string(self):
        cipher = encrypt_value("")
        assert decrypt_value(cipher) == ""


class TestPlaintextPassthrough:
    def test_plain_string(self):
        assert decrypt_value("hello world") == "hello world"

    def test_empty_string(self):
        assert decrypt_value("") == ""

    def test_none_like_non_string(self):
        # decrypt_value is typed str, but should not crash on odd input
        assert decrypt_value("not-encrypted") == "not-encrypted"


class TestIsEncrypted:
    def test_encrypted_value(self):
        cipher = encrypt_value("secret")
        assert is_encrypted(cipher) is True

    def test_plain_value(self):
        assert is_encrypted("plain") is False

    def test_prefix_only(self):
        assert is_encrypted("ENC:") is True

    def test_non_string(self):
        assert is_encrypted(42) is False  # type: ignore[arg-type]


# ── Master key file permissions ─────────────────────────────


class TestMasterKeyPermissions:
    def test_key_created_with_0600(self, _isolated_master_key):
        # Force key creation
        encrypt_value("trigger")
        key_path = _isolated_master_key
        assert key_path.exists()
        mode = key_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"

    def test_key_file_32_bytes(self, _isolated_master_key):
        encrypt_value("trigger")
        assert len(_isolated_master_key.read_bytes()) == 32


# ── Settings integration ────────────────────────────────────


class TestSettingsAutoDecrypt:
    def test_get_decrypts_enc_value(self, tmp_path):
        """Settings.get() transparently decrypts ENC: values."""
        from openclaw.config.settings import Settings

        cipher = encrypt_value("my-api-key")

        s = Settings.__new__(Settings)
        s._config = {"providers": {"anthropic": {"api_key": cipher}}}
        s._base_dir = tmp_path

        assert s.get("providers.anthropic.api_key") == "my-api-key"

    def test_get_returns_plain_value_unchanged(self, tmp_path):
        from openclaw.config.settings import Settings

        s = Settings.__new__(Settings)
        s._config = {"gateway": {"port": 18789}}
        s._base_dir = tmp_path

        assert s.get("gateway.port") == 18789


class TestSettingsAutoEncrypt:
    def test_set_encrypts_api_key(self, tmp_path):
        from openclaw.config.settings import Settings

        s = Settings.__new__(Settings)
        s._config = {}
        s._user_config_path = None
        s._base_dir = tmp_path

        s.set("providers.anthropic.api_key", "sk-ant-live-XYZ")

        raw = s._config["providers"]["anthropic"]["api_key"]
        assert raw.startswith("ENC:")
        # get() should transparently decrypt
        assert s.get("providers.anthropic.api_key") == "sk-ant-live-XYZ"

    def test_set_encrypts_token_key(self, tmp_path):
        from openclaw.config.settings import Settings

        s = Settings.__new__(Settings)
        s._config = {}
        s._user_config_path = None
        s._base_dir = tmp_path

        s.set("telegram.bot_token", "123456:ABC-DEF")

        raw = s._config["telegram"]["bot_token"]
        assert raw.startswith("ENC:")

    def test_set_encrypts_secret_key(self, tmp_path):
        from openclaw.config.settings import Settings

        s = Settings.__new__(Settings)
        s._config = {}
        s._user_config_path = None
        s._base_dir = tmp_path

        s.set("webhook.signing_secret", "whsec_abc")

        raw = s._config["webhook"]["signing_secret"]
        assert raw.startswith("ENC:")

    def test_set_skips_non_sensitive(self, tmp_path):
        from openclaw.config.settings import Settings

        s = Settings.__new__(Settings)
        s._config = {}
        s._user_config_path = None
        s._base_dir = tmp_path

        s.set("gateway.port", 9999)
        assert s._config["gateway"]["port"] == 9999

    def test_set_skips_already_encrypted(self, tmp_path):
        from openclaw.config.settings import Settings

        cipher = encrypt_value("already-encrypted")

        s = Settings.__new__(Settings)
        s._config = {}
        s._user_config_path = None
        s._base_dir = tmp_path

        s.set("providers.openai.api_key", cipher)

        raw = s._config["providers"]["openai"]["api_key"]
        # Should NOT double-encrypt
        assert raw == cipher

    def test_set_skips_empty_string(self, tmp_path):
        from openclaw.config.settings import Settings

        s = Settings.__new__(Settings)
        s._config = {}
        s._user_config_path = None
        s._base_dir = tmp_path

        s.set("providers.anthropic.api_key", "")

        raw = s._config["providers"]["anthropic"]["api_key"]
        assert raw == ""
