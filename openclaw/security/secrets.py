"""
Fernet-based encryption for secrets stored in user.yaml.

Master key lifecycle:
- On first launch a random 32-byte key is generated and saved to
  ``~/.openclaw/master.key`` (permissions 0600, outside the repo).
- The path is overridable via ``OPENCLAW_MASTER_KEY_PATH``.
- Inside Docker, ``entrypoint.sh`` writes the key from the env var
  ``OPENCLAW_MASTER_KEY`` so containers stay stateless.

Public API:
    encrypt_value(plaintext)  -> "ENC:<base64>"
    decrypt_value(ciphertext) -> plaintext  (passthrough if not encrypted)
    is_encrypted(value)       -> bool
"""

import base64
import logging
import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger("openclaw.security")

_PREFIX = "ENC:"

# Fixed salt — key uniqueness comes from the random master key itself.
_SALT = b"openclaw-fernet-v1"


def _default_key_path() -> Path:
    return Path(os.environ.get(
        "OPENCLAW_MASTER_KEY_PATH",
        Path.home() / ".openclaw" / "master.key",
    ))


def _derive_fernet_key(master_bytes: bytes) -> bytes:
    """Derive a URL-safe Fernet key from raw master key bytes via PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=480_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(master_bytes))


def _load_or_create_master_key() -> bytes:
    """Return the raw master key, creating one on first call."""
    key_path = _default_key_path()
    if key_path.exists():
        return key_path.read_bytes().strip()

    # Generate a fresh random key
    master = os.urandom(32)

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(master)
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600

    logger.info("Generated new master key at %s", key_path)
    return master


def _get_fernet() -> Fernet:
    """Build (and cache) a Fernet instance from the master key."""
    if not hasattr(_get_fernet, "_instance"):
        master = _load_or_create_master_key()
        fernet_key = _derive_fernet_key(master)
        _get_fernet._instance = Fernet(fernet_key)
    return _get_fernet._instance


def reset_fernet_cache() -> None:
    """Clear the cached Fernet instance (useful for tests)."""
    if hasattr(_get_fernet, "_instance"):
        del _get_fernet._instance


# ── Public API ──────────────────────────────────────────────


def is_encrypted(value: str) -> bool:
    """Return True if *value* carries the ``ENC:`` prefix."""
    return isinstance(value, str) and value.startswith(_PREFIX)


def encrypt_value(plaintext: str) -> str:
    """Encrypt *plaintext* and return ``"ENC:<base64>"``."""
    f = _get_fernet()
    token = f.encrypt(plaintext.encode("utf-8"))
    return _PREFIX + token.decode("ascii")


def decrypt_value(ciphertext: str) -> str:
    """Decrypt an ``"ENC:…"`` string.  Plain strings pass through unchanged."""
    if not is_encrypted(ciphertext):
        return ciphertext
    f = _get_fernet()
    token = ciphertext[len(_PREFIX):].encode("ascii")
    return f.decrypt(token).decode("utf-8")
