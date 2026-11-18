from __future__ import annotations

import io
import json
import unittest
import urllib.error

from boca_de_lobo.weather import Forecast, ForecastUnavailable, fetch_forecast


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode("utf-8") if not isinstance(payload, bytes) else payload

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(self._body)

    def __exit__(self, *exc_info: object) -> None:
        return None


def payload(values: list[object]) -> dict[str, object]:
    return {"hourly": {"time": ["t"] * len(values), "precipitation": values}}


class FetchForecastTests(unittest.TestCase):
    def test_returns_max_hourly_precipitation_ignoring_missing_values(self) -> None:
        opener = lambda url, timeout: FakeResponse(payload([0.2, None, 9.4, 1.0]))  # noqa: E731
        forecast = fetch_forecast(-23.5, -46.6, opener=opener)
        self.assertEqual(forecast, Forecast(max_hourly_mm=9.4, hours=3))
        self.assertTrue(forecast.is_strong_rain())

    def test_strong_rain_threshold_is_inclusive(self) -> None:
        self.assertTrue(Forecast(7.6, 24).is_strong_rain(7.6))
        self.assertFalse(Forecast(7.5, 24).is_strong_rain(7.6))

    def test_retries_with_exponential_backoff_then_succeeds(self) -> None:
        calls: list[int] = []
        sleeps: list[float] = []

        def opener(url: str, timeout: float) -> FakeResponse:
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.URLError("offline")
            return FakeResponse(payload([1.0]))

        forecast = fetch_forecast(0, 0, opener=opener, sleep=sleeps.append, backoff=1.0)
        self.assertEqual(forecast.max_hourly_mm, 1.0)
        self.assertEqual(sleeps, [1.0, 2.0])

    def test_raises_unavailable_after_all_attempts_fail(self) -> None:
        def opener(url: str, timeout: float) -> FakeResponse:
            raise TimeoutError("lento")

        with self.assertRaises(ForecastUnavailable):
            fetch_forecast(0, 0, opener=opener, sleep=lambda seconds: None, attempts=2)

    def test_malformed_payload_is_unavailable_without_retry(self) -> None:
        calls: list[int] = []

        def opener(url: str, timeout: float) -> FakeResponse:
            calls.append(1)
            return FakeResponse({"error": True})

        with self.assertRaises(ForecastUnavailable):
            fetch_forecast(0, 0, opener=opener, sleep=lambda seconds: None)
        self.assertEqual(len(calls), 1)

    def test_invalid_json_is_retried(self) -> None:
        calls: list[int] = []

        def opener(url: str, timeout: float) -> FakeResponse:
            calls.append(1)
            return FakeResponse(b"<html>")

        with self.assertRaises(ForecastUnavailable):
            fetch_forecast(0, 0, opener=opener, sleep=lambda seconds: None, attempts=3)
        self.assertEqual(len(calls), 3)

    def test_rejects_invalid_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            fetch_forecast(999, 0, opener=lambda url, timeout: FakeResponse(payload([1.0])))


if __name__ == "__main__":
    unittest.main()
