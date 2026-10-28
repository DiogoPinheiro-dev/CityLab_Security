"""Tecnicas de PDI do TCC, aplicadas ao frame antes da cadeia (docs/PLANO_TCC_PDI.md).

Cada tecnica recebe um frame BGR uint8 e devolve outro no mesmo formato e tipo.
Equalizacao, CLAHE e gama mexem so na luminancia (Y do YCrCb), para nao
alterar a cor; gaussiano e mascara de nitidez agem no frame BGR. Os parametros
padrao sao valores iniciais, a fixar no ensaio piloto (secao 3.5 do TCC).
"""
from typing import Any, Callable

import cv2
import numpy as np

# Valores iniciais de cada tecnica; uma configuracao pode trocar qualquer um.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "nenhuma": {},
    "equalizacao": {},
    "clahe": {"clip_limit": 2.0, "grid": 8},
    # Saida = 255 * (entrada / 255) ** gamma: abaixo de 1 clareia os escuros.
    "gama": {"gamma": 0.8},
    "gaussiano": {"ksize": 5, "sigma": 1.0},
    # Saida = frame + quantidade * (frame - frame suavizado).
    "nitidez": {"sigma": 1.0, "amount": 0.5},
}


def _on_luminance(frame: np.ndarray, transform: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = transform(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def neutral(frame: np.ndarray) -> np.ndarray:
    return frame


def global_equalization(frame: np.ndarray) -> np.ndarray:
    return _on_luminance(frame, cv2.equalizeHist)


def clahe(frame: np.ndarray, clip_limit: float, grid: int) -> np.ndarray:
    operator = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    return _on_luminance(frame, operator.apply)


def gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
    table = np.clip(np.round(255.0 * (np.arange(256) / 255.0) ** gamma), 0, 255).astype(np.uint8)
    return _on_luminance(frame, lambda channel: cv2.LUT(channel, table))


def gaussian(frame: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame, (ksize, ksize), sigma)


def unsharp(frame: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


TECHNIQUES: dict[str, Callable[..., np.ndarray]] = {
    "nenhuma": neutral,
    "equalizacao": global_equalization,
    "clahe": clahe,
    "gama": gamma,
    "gaussiano": gaussian,
    "nitidez": unsharp,
}


class Preprocessor:
    """Sequencia de tecnicas na ordem dada; a ordem faz parte da configuracao."""

    def __init__(self, steps: list[tuple[str, dict[str, Any]]]):
        if not steps:
            steps = [("nenhuma", {})]
        self.steps = []
        for name, params in steps:
            if name not in TECHNIQUES:
                raise ValueError(f"Tecnica de PDI desconhecida: {name}")
            unknown = set(params) - set(DEFAULT_PARAMS[name])
            if unknown:
                raise ValueError(f"Parametros desconhecidos em {name}: {sorted(unknown)}")
            self.steps.append((name, {**DEFAULT_PARAMS[name], **params}))

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        for name, params in self.steps:
            frame = TECHNIQUES[name](frame, **params)
        return frame

    @property
    def config(self) -> list[dict[str, Any]]:
        """Tecnicas, parametros efetivos e ordem, para o registro da rodada."""
        return [{"tecnica": name, **params} for name, params in self.steps]

    @classmethod
    def from_spec(cls, spec: str) -> "Preprocessor":
        """'clahe+gaussiano' ou 'gama:gamma=0.6+nitidez:amount=1'."""
        steps = []
        for part in filter(None, (item.strip() for item in spec.split("+"))):
            name, _, raw = part.partition(":")
            params = {}
            for pair in filter(None, raw.split(",")):
                key, _, value = pair.partition("=")
                params[key.strip()] = float(value) if "." in value else int(value)
            steps.append((name.strip(), params))
        return cls(steps)
