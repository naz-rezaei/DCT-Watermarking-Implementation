from collections.abc import Sequence
from math import isfinite
import random
import numpy as np
import cv2 as cv

from src.relative_rect import RelativeRect


class Attack:
    CUT = "Cut"
    COPY = "Copy"
    MOVE = "Move"
    ROTATE_X = "Rotate {:d} Degrees"
    FLIP_X = "Flip {:s}"
    SALT_AND_PEPPER_NOISE = "Salt and Pepper Noise"
    GAUSSIAN_NOISE = "Gaussian Noise"
    GAUSSIAN_FILTER = "Gaussian Filter"
    MEDIAN_FILTER = "Median Filter"
    HISTOGRAM_EQUALIZATION = "Histogram Equalization"
    JPEG_ENCODING = "JPEG Encoding"

    @staticmethod
    def validate_image(image: np.ndarray, image_name: str = "image"):
        """
        Checks an image to see if it is an 8-bit grayscale or an 8-bit RGB image.

        :param image: The image.
        :param image_name: The name to be used when raising errors, defaults to "image".
        :raises TypeError: If the image is not a numpy image.
        :raises ValueError: If the image is not an 8-bit grayscale or an 8-bit RGB image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(image_name + " is not a numpy image")
        if image.dtype != np.uint8:
            raise ValueError(image_name + " is not a valid 8-bit image")
        if image.ndim not in (2, 3) or image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("invalid shape of " + image_name)
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError("invalid number of channels in " + image_name)

    @staticmethod
    def validate_color(color: int | Sequence[int], color_name: str = "color"):
        """
        Checks a color to see if it is an integer value or a tuple or a list of 3 integer values.

        :param color: The color.
        :param color_name: The name to be used when raising errors, defaults to "color".
        :raises TypeError: If the color is not an integer or a tuple of integers or a list of integers.
        :raises ValueError: If any values exceeds the interval [0, 255].
        """
        if isinstance(color, int):
            if color < 0 or color > 255:
                raise ValueError("invalid channel value in " + color_name)
            return
        if not isinstance(color, (tuple, list)):
            raise TypeError("invalid " + color_name)
        if len(color) != 3:
            raise ValueError("invalid number of channels in " + color_name)
        for c in color:
            if not isinstance(c, int):
                raise TypeError("invalid channel value in " + color_name)
            if c < 0 or c > 255:
                raise ValueError("invalid channel value in " + color_name)

    @staticmethod
    def validate_color_for_image(color: int | Sequence[int], image: np.ndarray,
                                 color_name: str = "color", image_name: str = "image"):
        """
        Checks to see whether a color's structure matches an image's pixels' structures.

        Note
        ----
        The color and the image both must be validated before calling this method.

        :param color: The color.
        :param image: The image.
        :param color_name: The name to be used when raising errors, defaults to "color".
        :param image_name: The name to be used when raising errors, defaults to "image".
        :raises ValueError: If the color and the image do not match.
        """
        if image.ndim == 2 and not isinstance(color, int):
            raise ValueError(f"{image_name} is grayscale, but {color_name} is not")
        if image.ndim == 3 and not isinstance(color, (tuple, list)):
            raise ValueError(f"{image_name} is color, but {color_name} is not")

    @staticmethod
    def color_to_pixel(color: int | Sequence[int]) -> np.ndarray:
        return np.array(color, dtype=np.uint8)

    @property
    def type(self) -> str:
        return self.__type

    def apply(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def __init__(self, _type: str) -> None:
        if not isinstance(_type, str):
            raise ValueError("invalid attack type")
        self.__type = _type


class RegionTransportAttack(Attack):
    CANVAS_COLOR_NAME = "canvas color"

    @staticmethod
    def _validate_relative_rect(rect: RelativeRect, rect_name: str = "rect"):
        if not isinstance(rect, RelativeRect):
            raise TypeError(f"invalid '{rect_name}'")
        if not 0 <= rect.x <= 1 or not 0 <= rect.y <= 1 \
                or not 0 <= rect.x + rect.w <= 1 or not 0 <= rect.y + rect.h <= 1:
            raise ValueError(f"invalid boundaries for '{rect_name}'")

    @staticmethod
    def _clip_rect_in_image(image_width: int, image_height: int, x: int, y: int, w: int, h: int) \
            -> tuple[int, int, int, int]:
        x2 = x + w
        y2 = y + h
        if x2 < x:
            x, x2 = x2, x
        if y2 < y:
            y, y2 = y2, y
        x = min(max(x, 0), image_width)
        y = min(max(y, 0), image_height)
        x2 = min(max(x2, x), image_width)
        y2 = min(max(y2, y), image_height)
        return x, y, x2 - x, y2 - y
    
    def __init__(self, _type: str) -> None:
        super().__init__(_type)


class CutAttack(RegionTransportAttack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        Attack.validate_color_for_image(self.__canvas_color, image,
                                        RegionTransportAttack.CANVAS_COLOR_NAME)
        image = image.copy()
        x, y, w, h = self.__where.translate_toi(0, 0, image.shape[1], image.shape[0])
        x, y, w, h = RegionTransportAttack._clip_rect_in_image(
            image.shape[1], image.shape[0], x, y, w, h
        )
        if w > 0 and h > 0:
            image[y:y + h, x:x + w] = np.full((h, w), self.__canvas_color, np.uint8)
        return image
    
    def __init__(self, where: RelativeRect, canvas_color: int | Sequence[int]) -> None:
        super().__init__(Attack.CUT)
        RegionTransportAttack._validate_relative_rect(where)
        Attack.validate_color(canvas_color, RegionTransportAttack.CANVAS_COLOR_NAME)
        self.__where: RelativeRect = where.copy()
        self.__canvas_color = canvas_color


class CopyAttack(RegionTransportAttack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        image = image.copy()
        sx, sy, sw, sh = self.__source.translate_toi(0, 0, image.shape[1], image.shape[0])
        sx, sy, sw, sh = RegionTransportAttack._clip_rect_in_image(
            image.shape[1], image.shape[0], sx, sy, sw, sh
        )
        dx, dy, dw, dh = self.__destination.translate_toi(0, 0, image.shape[1], image.shape[0])
        dx, dy, dw, dh = RegionTransportAttack._clip_rect_in_image(
            image.shape[1], image.shape[0], dx, dy, dw, dh
        )
        if sw <= 0 or sh <= 0:
            return image
        if sw != dw or sh != dh:
            raise ValueError("source size and destination size mismatch")
        image[dy:dy + dh, dx:dx + dw] = image[sy:sy + sh, sx:sx + sw].copy()
        return image
    
    def __init__(self, source: RelativeRect, destination: RelativeRect) -> None:
        super().__init__(Attack.COPY)
        RegionTransportAttack._validate_relative_rect(source)
        RegionTransportAttack._validate_relative_rect(destination)
        self.__source: RelativeRect = source
        self.__destination: RelativeRect = destination


class MoveAttack(RegionTransportAttack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        Attack.validate_color_for_image(self.__canvas_color, image,
                                        RegionTransportAttack.CANVAS_COLOR_NAME)
        image = image.copy()
        sx, sy, sw, sh = self.__source.translate_toi(0, 0, image.shape[1], image.shape[0])
        sx, sy, sw, sh = RegionTransportAttack._clip_rect_in_image(
            image.shape[1], image.shape[0], sx, sy, sw, sh
        )
        dx, dy, dw, dh = self.__destination.translate_toi(0, 0, image.shape[1], image.shape[0])
        dx, dy, dw, dh = RegionTransportAttack._clip_rect_in_image(
            image.shape[1], image.shape[0], dx, dy, dw, dh
        )
        if sw <= 0 or sh <= 0:
            return image
        if sw != dw or sh != dh:
            raise ValueError("source size and destination size mismatch")
        src = image[sy:sy + sh, sx:sx + sw].copy()
        image[sy:sy + sh, sx:sx + sw] = np.full((sh, sw), self.__canvas_color, np.uint8)
        image[dy:dy + dh, dx:dx + dw] = src
        return image
    
    def __init__(self, source: RelativeRect, destination: RelativeRect,
                 canvas_color: int | Sequence[int]) -> None:
        super().__init__(Attack.MOVE)
        RegionTransportAttack._validate_relative_rect(source)
        RegionTransportAttack._validate_relative_rect(destination)
        Attack.validate_color(canvas_color, RegionTransportAttack.CANVAS_COLOR_NAME)
        self.__source: RelativeRect = source
        self.__destination: RelativeRect = destination
        self.__canvas_color = canvas_color


class RotateAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        image = image.copy()
        return image if self.__degrees == 0 else np.rot90(image, self.__degrees // 90, axes=(0, 1))
    
    @staticmethod
    def __validate_degrees(degrees: int) -> int:
        if not isinstance(degrees, (int, float)):
            raise TypeError("invalid degrees")
        if not isfinite(degrees):
            raise ValueError("invalid degrees")
        if degrees != int(degrees):
            raise ValueError("non-integer degrees")
        degrees = int(degrees) % 360
        if (degrees % 90) != 0:
            raise ValueError("degrees must be divisible by 90")
        return degrees

    def __init__(self, degrees: int) -> None:
        degrees = RotateAttack.__validate_degrees(degrees)
        super().__init__(Attack.ROTATE_X.format(degrees))
        self.__degrees: int = degrees


class FlipAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        image = image.copy()
        return np.flip(image, axis=0 if self.__vertical else 1)

    def __init__(self, vertical: bool) -> None:
        vertical = bool(vertical)
        super().__init__(Attack.FLIP_X.format("Vertical" if vertical else "Horizontal"))
        self.__vertical: bool = vertical


class SaltAndPepperNoiseAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        image = image.copy()
        black, white = (np.array(0, dtype=np.uint8), np.array(255, dtype=np.uint8)) if image.ndim == 2 \
            else (np.array((0, 0, 0), dtype=np.uint8), np.array((255, 255, 255), dtype=np.uint8))
        for _ in range(int(image.shape[1] * image.shape[0] * self.__density)):
            x, y = random.randrange(image.shape[1]), random.randrange(image.shape[0])
            image[y, x] = black if random.randrange(2) else white
        return image
        
    @staticmethod
    def __validate_density(density: int) -> float:
        if not isinstance(density, (int, float)):
            raise TypeError("invalid density")
        if not isfinite(density) or density < 0 or density > 1:
            raise ValueError("invalid density")
        return float(density)

    def __init__(self, density: float) -> None:
        super().__init__(Attack.SALT_AND_PEPPER_NOISE)
        self.__density: float = SaltAndPepperNoiseAttack.__validate_density(density)


class GaussianNoiseAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        image = image.copy()
        gaussian = np.random.normal(self.__mean, self.__variance ** 0.5, image.shape)
        return np.clip(image.astype(gaussian.dtype) + gaussian, 0, 255).astype(np.uint8)
        
    @staticmethod
    def __validate_mean_and_variance(mean: float, variance: float):
        if not isinstance(mean, (int, float)) or not isinstance(variance, (int, float)):
            raise TypeError("invalid mean/variance")
        if not isfinite(mean) or not isfinite(variance) or variance <= 0:
            raise ValueError("invalid mean/variance")

    def __init__(self, mean: float, variance: float) -> None:
        super().__init__(Attack.GAUSSIAN_NOISE)
        GaussianNoiseAttack.__validate_mean_and_variance(mean, variance)
        self.__mean: float = float(mean)
        self.__variance: float = float(variance)


class GaussianFilterAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        return cv.GaussianBlur(image, ksize=(self.__kernel_size, self.__kernel_size), sigmaX=0, sigmaY=0)
        
    @staticmethod
    def __validate_kernel_size(kernel_size: int):
        if not isinstance(kernel_size, int):
            raise TypeError("invalid kernel-size")
        if kernel_size < 3 or (kernel_size % 2) == 0:
            raise ValueError("invalid kernel-size")

    def __init__(self, kernel_size: int) -> None:
        super().__init__(Attack.GAUSSIAN_FILTER)
        GaussianFilterAttack.__validate_kernel_size(kernel_size)
        self.__kernel_size: int = kernel_size


class MedianFilterAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        return cv.medianBlur(image, ksize=self.__kernel_size)
        
    @staticmethod
    def __validate_kernel_size(kernel_size: int):
        if not isinstance(kernel_size, int):
            raise TypeError("invalid kernel-size")
        if kernel_size < 3 or (kernel_size % 2) == 0:
            raise ValueError("invalid kernel-size")

    def __init__(self, kernel_size: int) -> None:
        super().__init__(Attack.MEDIAN_FILTER)
        MedianFilterAttack.__validate_kernel_size(kernel_size)
        self.__kernel_size: int = kernel_size


class HistogramEqualizationAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        Attack.validate_image(image)
        return cv.equalizeHist(image)
        
    def __init__(self) -> None:
        super().__init__(Attack.HISTOGRAM_EQUALIZATION)


class JPEGEncodingAttack(Attack):

    def apply(self, image: np.ndarray) -> np.ndarray:
        buffer = cv.imencode(".jpg", image, (cv.IMWRITE_JPEG_QUALITY, self.__quality))[1]
        return cv.imdecode(np.frombuffer(buffer, np.uint8), -1)
             
    @staticmethod
    def __validate_quality(quality: int):
        if not isinstance(quality, int):
            raise TypeError("invalid quality")
        if quality < 0 or quality > 100:
            raise ValueError("invalid quality")
   
    def __init__(self, quality: int) -> None:
        super().__init__(Attack.JPEG_ENCODING)
        JPEGEncodingAttack.__validate_quality(quality)
        self.__quality: int = quality
