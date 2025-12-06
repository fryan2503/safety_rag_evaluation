
import base64
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import uuid
from zoneinfo import ZoneInfo


def now_et() -> str:
    """Return current ET timestamp."""
    return datetime.now(ZoneInfo("America/New_York")).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def read_text(maybe_path: Optional[str]) -> str:
    """Load from file or return raw string."""
    if maybe_path is None:
        return ""
    p = Path(maybe_path)
    return p.read_text(encoding="utf-8") if p.exists() else maybe_path


def make_permutation_id(metadata: Dict[str, Any]) -> str:
    """Create reversible, URL-safe experiment ID."""
    payload = {
        "metadata": metadata,
        "run_uuid": str(uuid.uuid4()),
    }

    json_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(json_bytes).digest()

    combined = json_bytes + digest[:8]
    encoded = base64.urlsafe_b64encode(combined).decode("ascii").rstrip("=")

    return encoded


def parse_permutation_id(pid: str, return_json: bool = False) -> Dict[str, Any]:
    """Decode & verify a permutation_id created by make_permutation_id()."""
    padding = "=" * (-len(pid) % 4)
    decoded = base64.urlsafe_b64decode(pid + padding)

    json_bytes, digest_suffix = decoded[:-8], decoded[-8:]
    expected_digest = hashlib.sha256(json_bytes).digest()[:8]

    if digest_suffix != expected_digest:
        raise ValueError("Integrity check failed — ID may be corrupted.")

    payload_json = json_bytes.decode("utf-8")

    return payload_json if return_json else json.loads(payload_json)