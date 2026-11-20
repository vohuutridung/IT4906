"""Live progress for stages that may block without emitting their own logs."""

from collections.abc import Iterator
from contextlib import contextmanager
from threading import Event, Thread
from time import monotonic


@contextmanager
def progress_stage(label: str, *, interval: float = 15.0) -> Iterator[None]:
    start = monotonic()
    stop = Event()

    def heartbeat() -> None:
        while not stop.wait(interval):
            print(f"{label} · chưa hoàn tất ({monotonic() - start:.0f}s)", flush=True)

    print(f"{label} · bắt đầu", flush=True)
    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    outcome = "dừng"
    try:
        yield
        outcome = "xong"
    finally:
        stop.set()
        thread.join()
        print(f"{label} · {outcome} ({monotonic() - start:.1f}s)", flush=True)
