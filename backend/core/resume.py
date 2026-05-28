import logging
from pathlib import Path

import tiktoken

log = logging.getLogger(__name__)
_enc = tiktoken.get_encoding("cl100k_base")
_MAX_TOKENS = 8000


def load_resume(path: Path) -> str:
    if not path.exists():
        log.warning("Resume file not found: %s (fallback mode will indicate empty resume)", path)
        return ""
    text = path.read_text(encoding="utf-8")
    tok = _enc.encode(text)
    if len(tok) > _MAX_TOKENS:
        log.warning("Resume too long (%d tokens), truncating to %d", len(tok), _MAX_TOKENS)
        text = _enc.decode(tok[:_MAX_TOKENS])
    return text
