"""
session_marker.py - non-blocking keypress marker logger.

This is a module, not a script.  Import and use MarkerLogger from receiver.py.
Timestamps are ESP32 millis() values from the last decoded packet - not host
wall-clock time - so all labels live in the same clock domain as the raw data.
"""
import csv
import sys
import os

# Keypress -> BFRB/tic event label. Keys not listed here are consumed and
# ignored (no marker written) so stray keystrokes don't pollute the CSV.
KEY_EVENTS = {
    "[": "skin_picking_hand",
    "]": "lip_picking",
    "\\": "skin_picking_face",
}


def _make_nonblocking_check():
    """Return a platform-appropriate non-blocking stdin-check function.

    Returns a callable that returns True if a key was pressed, False otherwise.
    """
    if sys.platform == "win32":
        import msvcrt

        def _check() -> bool:
            return msvcrt.kbhit()

        return _check
    else:
        import select

        def _check() -> bool:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            return bool(ready)

        return _check


def _make_key_reader():
    """Return a platform-appropriate function that reads and returns one keypress."""
    if sys.platform == "win32":
        import msvcrt

        def _read() -> str:
            return msvcrt.getwch()

        return _read
    else:
        def _read() -> str:
            return sys.stdin.read(1)

        return _read


class MarkerLogger:
    """Write ESP32-timestamped marker rows to a CSV file.

    Usage (inside receiver's read loop)::

        marker = MarkerLogger("data/raw/session_001.markers.csv")
        while reading:
            pkt = decode_next_packet(...)
            marker.check_and_log(pkt.timestamp_ms)
    """

    def __init__(self, path: str):
        self._path = path
        self._check_key = _make_nonblocking_check()
        self._read_key = _make_key_reader()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["timestamp_ms", "event"])
        self._fh.flush()

    def check_and_log(self, last_ts_ms: int) -> bool:
        """Call once per receiver loop iteration.

        If a key in KEY_EVENTS was pressed, writes *last_ts_ms* (the ESP32
        timestamp of the most recently decoded packet) along with the mapped
        event label to the CSV. Any other keypress is consumed and ignored.

        Returns True if a marker was written.
        """
        if not self._check_key():
            return False
        # Always consume the keypress so it doesn't accumulate, even if unmapped
        key = self._read_key()

        event = KEY_EVENTS.get(key)
        if event is None:
            return False

        self._writer.writerow([last_ts_ms, event])
        self._fh.flush()
        return True

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
