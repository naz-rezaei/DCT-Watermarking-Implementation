from typing import Self
from math import isfinite
import random


class RelativeRect:

    @staticmethod
    def __validate_param(param, param_name: str):
        if not isinstance(param, (int, float)):
            raise TypeError(f"invalid {param_name}")
        if not isfinite(param):
            raise ValueError(f"invalid {param_name}")

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def w(self) -> float:
        return self.__w

    @property
    def h(self) -> float:
        return self.__h

    def copy(self) -> Self:
        return RelativeRect(self.__x, self.__y, self.__w, self.__h)
    
    def translate_to(self, x: float, y: float, w: float, h: float) \
            -> tuple[float, float, float, float]:
        RelativeRect.__validate_param(x, "x")
        RelativeRect.__validate_param(y, "y")
        RelativeRect.__validate_param(w, "w")
        RelativeRect.__validate_param(h, "h")
        _w = self.__w * w
        _h = self.__h * h
        x += self.__x * w
        y += self.__y * h
        return x, y, _w, _h

    def translate_toi(self, x: float, y: float, w: float, h: float) \
            -> tuple[int, int, int, int]:
        RelativeRect.__validate_param(x, "x")
        RelativeRect.__validate_param(y, "y")
        RelativeRect.__validate_param(w, "w")
        RelativeRect.__validate_param(h, "h")
        _w = self.__w * w
        _h = self.__h * h
        x += self.__x * w
        y += self.__y * h
        return round(x), round(y), round(_w), round(_h)

    def __str__(self) -> str:
        return f"({round(self.__x, 6)}, {round(self.__y, 6)}), ({round(self.__w, 6)}x{round(self.__h, 6)})"
    
    def __init__(self, x: float, y: float, w: float, h: float) -> None:
        RelativeRect.__validate_param(x, "x")
        RelativeRect.__validate_param(y, "y")
        RelativeRect.__validate_param(w, "w")
        RelativeRect.__validate_param(h, "h")
        self.__x = float(x)
        self.__y = float(y)
        self.__w = float(w)
        self.__h = float(h)


def _validate_interval_param(param, param_name: str):
    if not isinstance(param, (tuple, list)):
        raise TypeError(f"invalid {param_name}")
    if len(param) != 2 \
            or not isinstance(param[0], (int, float)) or not isinstance(param[1], (int, float)) \
            or not isfinite(param[0]) or not isfinite(param[1]):
        raise ValueError(f"invalid {param_name}")
    if param[1] < param[0]:
        raise ValueError(f"invalid boundaries of {param_name}")


def _random_relative_rects_validate(interval_hor_ex: tuple[float, float],
                                    interval_ver_ex: tuple[float, float],
                                    interval_w: tuple[float, float], interval_h: tuple[float, float],
                                    max_n: int):
    _validate_interval_param(interval_hor_ex, "interval horizontal extent")
    _validate_interval_param(interval_ver_ex, "interval vertical extent")
    _validate_interval_param(interval_w, "interval w")
    _validate_interval_param(interval_h, "interval h")
    if not isinstance(max_n, int):
        raise TypeError("invalid max n")
    if max_n < 1:
        raise ValueError("invalid max n")


def _random_relative_rects_generate_size(interval_w: tuple[float, float],
                                         interval_h: tuple[float, float]):
    w = interval_w[0] if interval_w[0] == interval_w[1] \
        else interval_w[0] + random.random() * (interval_w[1] - interval_w[0])
    h = interval_h[0] if interval_h[0] == interval_h[1] \
        else interval_h[0] + random.random() * (interval_h[1] - interval_h[0])
    return w, h


def random_relative_rects(interval_hor_ex: tuple[float, float], interval_ver_ex: tuple[float, float],
                          interval_w: tuple[float, float], interval_h: tuple[float, float],
                          overlap: bool = False, all_of_same_size: bool = False,
                          max_n: int = 1) -> list[RelativeRect]:
    # Preperation
    _random_relative_rects_validate(interval_hor_ex, interval_ver_ex, interval_w, interval_h, max_n)
    all_of_same_size = bool(all_of_same_size)
    overlap = bool(overlap)
    # Initialization
    answer = []
    hor_ex_es = [tuple(interval_hor_ex)]  # All possible horizontal extents
    ver_ex_es = [tuple(interval_hor_ex)]  # All possible vertical extents
    w, h = _random_relative_rects_generate_size(interval_w, interval_h) if all_of_same_size else (0, 0)
    n = 0
    while n < max_n:
        # Size of the current rect
        if not all_of_same_size:
            w, h = _random_relative_rects_generate_size(interval_w, interval_h)
        # The extents (including their indexes) in which the rect can fit
        fit_hor_ex_es = [(i, ex) for i, ex in enumerate(hor_ex_es) if w <= (ex[1] - ex[0])]
        fit_ver_ex_es = [(i, ex) for i, ex in enumerate(ver_ex_es) if h <= (ex[1] - ex[0])]
        if not fit_hor_ex_es or not fit_ver_ex_es:
            break  # Nowhere to place the rect
        # Select extents
        hor_ex_i = random.randrange(len(fit_hor_ex_es))
        ver_ex_i = random.randrange(len(fit_ver_ex_es))
        hor_ex, ver_ex = fit_hor_ex_es[hor_ex_i][1], fit_ver_ex_es[ver_ex_i][1]
        # Build the coordinates in the selected extents and add the rect
        x = hor_ex[0] + random.random() * (hor_ex[1] - hor_ex[0] - w)
        y = ver_ex[0] + random.random() * (ver_ex[1] - ver_ex[0] - h)
        answer.append(RelativeRect(x, y, w, h))
        n += 1
        # Split selected extents if 'overlap' is disabled
        if not overlap:
            # Hor
            i = fit_hor_ex_es[hor_ex_i][0]
            hor_ex = hor_ex_es[i]
            del hor_ex_es[i]
            hor_ex_es.insert(i, (hor_ex[0], x))
            hor_ex_es.insert(i + 1, (x + w, hor_ex[1]))
            # Ver
            i = fit_ver_ex_es[ver_ex_i][0]
            ver_ex = ver_ex_es[i]
            del ver_ex_es[i]
            ver_ex_es.insert(i, (ver_ex[0], y))
            ver_ex_es.insert(i + 1, (y + h, ver_ex[1]))
    return answer
