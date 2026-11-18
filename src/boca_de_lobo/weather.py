"""Previsao de chuva (Open-Meteo, sem chave de API) com tentativas e falha explicita."""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger("boca_de_lobo.weather")

API_URL = "https://api.open-meteo.com/v1/forecast"
# Limiar padrao de "chuva forte" em mm/h; configuravel na automacao.
STRONG_RAIN_MM_PER_HOUR = 7.6


class ForecastUnavailable(Exception):
    """A previsao nao pode ser obtida; nunca deve ser tratada como 'sem chuva'."""


@dataclass(frozen=True)
class Forecast:
    max_hourly_mm: float
    hours: int
    source: str = "open-meteo"

    def is_strong_rain(self, threshold: float = STRONG_RAIN_MM_PER_HOUR) -> bool:
        return self.max_hourly_mm >= threshold


def _parse(payload: Any, hours: int) -> Forecast:
    try:
        values = payload["hourly"]["precipitation"]
    except (KeyError, TypeError) as error:
        raise ForecastUnavailable("Resposta sem hourly.precipitation.") from error
    numbers = [
        float(value)
        for value in values[:hours]
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not numbers:
        raise ForecastUnavailable("Resposta sem valores numericos de precipitacao.")
    return Forecast(max_hourly_mm=max(numbers), hours=len(numbers))


def fetch_forecast(
    latitude: float,
    longitude: float,
    *,
    hours: int = 24,
    attempts: int = 3,
    timeout: float = 10.0,
    backoff: float = 1.0,
    opener: Callable[..., Any] = urllib.request.urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> Forecast:
    """Maior precipitacao horaria prevista nas proximas `hours` horas."""
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError("Latitude/longitude fora do intervalo valido.")
    if hours <= 0 or attempts <= 0:
        raise ValueError("hours e attempts devem ser positivos.")

    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "precipitation",
            "forecast_hours": hours,
            "timezone": "auto",
        }
    )
    url = f"{API_URL}?{query}"
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with opener(url, timeout=timeout) as response:
                return _parse(json.load(response), hours)
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            LOGGER.warning("Previsao: tentativa %d/%d falhou (%s).", attempt, attempts, error)
            if attempt < attempts:
                sleep(backoff * 2 ** (attempt - 1))
    raise ForecastUnavailable(f"API indisponivel apos {attempts} tentativas: {last_error}") from last_error
