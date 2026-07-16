"""
session_marker.py — non-blocking keypress marker logger.

This is a module, not a script.  Import and use MarkerLogger from receiver.py.
Timestamps are ESP32 millis() values from the last decoded packet — not host
wall-clock time — so all labels live in the same clock domain as the raw data.
"""
import csv
import sys
import os


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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["timestamp_ms", "event"])
        self._fh.flush()

    def check_and_log(self, last_ts_ms: int, event: str = "marker") -> bool:
        """Call once per receiver loop iteration.

        If a key was pressed, writes *last_ts_ms* (the ESP32 timestamp of the
        most recently decoded packet) along with *event* to the CSV.

        Returns True if a marker was written.
        """
        if not self._check_key():
            return False
        # Consume the keypress so it doesn't accumulate
        if sys.platform == "win32":
            import msvcrt
            msvcrt.getwch()
        else:
            sys.stdin.read(1)

        self._writer.writerow([last_ts_ms, event])
        self._fh.flush()
        return True

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
