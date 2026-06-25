"""
modules/audio_capture.py
────────────────────────
Captures live microphone audio in fixed-duration segments using SoundDevice.
Each captured segment is a NumPy float32 array at SAMPLE_RATE Hz.
"""

import queue
import time
import warnings

import numpy as np
import sounddevice as sd

from src.core.config import SAMPLE_RATE, CHANNELS, SEGMENT_DURATION


def _find_input_device() -> int | None:
    """Return the device index of the best available input device.

    Priority order:
    1. The system default input device (marked with '>' in sd.query_devices())
    2. First device that has ≥1 input channel and 'microphone' in its name (case-insensitive)
    3. First device that has ≥1 input channel
    4. None  →  let sounddevice pick (may fail, but gives a clear error)
    """
    try:
        # sd.default.device is a (input_idx, output_idx) tuple
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            info = sd.query_devices(default_in)
            if info["max_input_channels"] >= 1:
                return int(default_in)
    except Exception:
        pass

    # Fallback: scan all devices
    devices = sd.query_devices()
    mic_idx = None
    first_input_idx = None
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        if first_input_idx is None:
            first_input_idx = idx
        if "microphone" in dev["name"].lower() and mic_idx is None:
            mic_idx = idx

    return mic_idx if mic_idx is not None else first_input_idx


class AudioCapture:
    """Continuously records audio from the system microphone.

    Usage
    -----
    capture = AudioCapture()
    capture.start()
    segment = capture.get_segment()   # blocks until a segment is ready
    capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        segment_duration: float = SEGMENT_DURATION,
        device: int | None = None,
    ) -> None:
        self.sample_rate     = sample_rate
        self.channels        = channels
        self.segment_samples = int(sample_rate * segment_duration)

        # Resolve device: explicit arg → auto-detect → None (sd picks)
        self.device = device if device is not None else _find_input_device()

        # Block size: 512 samples is a safe, low-latency default on macOS
        self._blocksize = 512

        # Lock-free buffer queue containing raw audio blocks from the callback
        self._block_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=300)
        self._buffer: list[np.ndarray]   = []
        self._stream: sd.InputStream | None = None
        self._running = False

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Open the input stream and begin buffering."""
        self._running = True
        self._buffer  = []
        
        # Clear any stale blocks in the queue
        while not self._block_queue.empty():
            try:
                self._block_queue.get_nowait()
            except queue.Empty:
                break

        kwargs: dict = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        if self.device is not None:
            kwargs["device"] = self.device

        self._stream = sd.InputStream(**kwargs)
        self._stream.start()

    def stop(self) -> None:
        """Stop recording and close the stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_segment(self, timeout: float = 10.0) -> np.ndarray | None:
        """Block until a full segment is available, then return it.

        Returns None if the timeout expires or recording has stopped.
        Handles backlog pruning to prevent stale audio latency build-up.
        """
        start_time = time.time()
        
        # Prune any backlog of old audio blocks to maintain real-time sync
        self._prune_backlog()

        while self._running:
            # Check if we have accumulated enough samples for a segment
            total = sum(len(b) for b in self._buffer)
            if total >= self.segment_samples:
                combined  = np.concatenate(self._buffer)
                segment   = combined[: self.segment_samples]
                remainder = combined[self.segment_samples :]
                self._buffer = [remainder] if len(remainder) else []
                return segment

            # Calculate remaining timeout
            elapsed = time.time() - start_time
            rem_timeout = timeout - elapsed
            if rem_timeout <= 0:
                return None

            try:
                # Wait for next block
                block = self._block_queue.get(timeout=rem_timeout)
                self._buffer.append(block)
            except queue.Empty:
                return None

        return None

    @property
    def active_device_name(self) -> str:
        """Human-readable name of the device being used."""
        if self.device is None:
            return "system default"
        try:
            return sd.query_devices(self.device)["name"]
        except Exception:
            return f"device #{self.device}"

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _prune_backlog(self) -> None:
        """Discard older blocks if the queue size exceeds our maximum buffer size."""
        # Allow at most 1.5 segments worth of blocks (approx 140 blocks) to prevent latency
        max_buffered_blocks = int((self.segment_samples / self._blocksize) * 1.5)
        
        drained = 0
        while self._block_queue.qsize() > max_buffered_blocks:
            try:
                self._block_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
                
        if drained > 0:
            # If we skipped ahead in the stream, discard the current partial buffer
            # to prevent audio discontinuity glitches
            self._buffer.clear()

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """SoundDevice callback – accumulates samples into fixed segments.
        
        Runs on PortAudio's real-time thread. Must be extremely fast and lock-free.
        """
        if not self._running:
            return

        # Log any xrun/overflow warnings without crashing
        if status and status.input_overflow:
            warnings.warn(
                f"[AudioCapture] Input overflow — consider increasing blocksize "
                f"(current: {self._blocksize})",
                stacklevel=2,
            )

        try:
            # Put the raw block directly into the queue.
            # If queue is full, drop block. The consumer's prune_backlog will recover.
            self._block_queue.put_nowait(indata.copy().flatten())
        except queue.Full:
            pass
