from collections.abc import Iterable
import cv2 as cv


def cv_wait_for_esc_or_x(window_names: str | Iterable[str]):
    # One window
    if isinstance(window_names, str):
        while cv.waitKey(10) & 0xFF != 27 \
                and int(cv.getWindowProperty(window_names, cv.WND_PROP_VISIBLE)) > 0:
            pass
        return
    # Multiple windows
    while cv.waitKey(10) & 0xFF != 27:
        all_closed = True
        for window_name in window_names:
            if int(cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE)) > 0:
                all_closed = False
                break
        if all_closed:
            break


def cv_destroy_windows_if_shown(window_names: str | Iterable[str]):
    if isinstance(window_names, str):
        window_names = window_names,
    for window_name in window_names:
        if int(cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE)) > 0:
            cv.destroyWindow(window_name)


def hash_length_in_bytes(hash_length: int) -> int:
    """Returns the number of necessary bytes for storage of given number of bits of hash-length."""
    return int(hash_length / 8) if hash_length % 8 == 0 else int(hash_length / 8) + 1


def bytes_to_hex(_bytes: bytes | bytearray) -> str:
    s = ""
    for b in _bytes:
        s += f"{b:02X}"
    return s


def calculate_hash_length(cell_size: int, lsb_n: int) -> int:
    return max(1, min(256, int(cell_size ** 2 * lsb_n)))
