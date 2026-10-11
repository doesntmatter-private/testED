"""Alert outputs. Uses Jetson.GPIO when present, otherwise logs."""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger("laneguard.alerts")

try:
    import Jetson.GPIO as GPIO  # type: ignore
    HAVE_GPIO = True
except ImportError:
    GPIO = None
    HAVE_GPIO = False


class Alerter:
    def __init__(self, relay_pin: int = 7, pulse_s: float = 3.0):
        self.relay_pin = relay_pin
        self.pulse_s = pulse_s
        self._lock = threading.Lock()
        if HAVE_GPIO:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(relay_pin, GPIO.OUT, initial=GPIO.LOW)
            log.info("GPIO relay on BOARD pin %d", relay_pin)
        else:
            log.info("Jetson.GPIO not available, alerts will be logged only")

    def fire(self, event) -> None:
        log.warning("ALARM track=%d t=%.2f anchor=%s speed=%.1fpx/f",
                    event.track_id, event.t, tuple(round(v) for v in event.anchor), event.speed_px_per_frame)
        if HAVE_GPIO:
            threading.Thread(target=self._pulse, daemon=True).start()

    def _pulse(self) -> None:
        with self._lock:
            GPIO.output(self.relay_pin, GPIO.HIGH)
            time.sleep(self.pulse_s)
            GPIO.output(self.relay_pin, GPIO.LOW)

    def close(self) -> None:
        if HAVE_GPIO:
            GPIO.cleanup()
