"""Atomic artifact writes; existing fixed splits are never overwritten."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
from typing import TextIO


@contextmanager
def new_text_file(path: str | Path) -> Iterator[TextIO]:
    """Publish a complete UTF-8 file, failing if the destination exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    # mkstemp always creates 0600, which on our ACL-shared scratch dir zeroes the ACL mask and locks
    # teammates out. Give the file the mode a plain open() would have (0666 minus the umask).
    umask = os.umask(0)
    os.umask(umask)
    try:
        os.chmod(temporary, 0o666 & ~umask)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            yield stream
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
