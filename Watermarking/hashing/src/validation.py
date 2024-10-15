import numpy as np

from src.misc import hash_length_in_bytes


def validate_arnold_iterations(arnold_iterations: int):
    if not isinstance(arnold_iterations, int):
        raise TypeError("invalid Arnold-iterations")
    if arnold_iterations < 0:
        raise ValueError("invalid Arnold-iterations")


def validate_block_and_cell_size(block_size: int, cell_size: int):
    if not isinstance(block_size, int):
        raise TypeError("invalid block-size")
    if not isinstance(cell_size, int):
        raise TypeError("invalid cell-size")
    if block_size < 2 or block_size > 512:
        raise ValueError("block-size must be in range [2, 512]")
    if cell_size < 1 or cell_size > 16:
        raise ValueError("cell-size must be in range [1, 16]")
    if cell_size > block_size:
        raise ValueError("cell-size must be smaller than or equal to block-size")


def validate_hash_length(hash_length: int):
    if not isinstance(hash_length, int):
        raise TypeError("invalid hash length")
    if hash_length < 1 or hash_length > 256:
        raise ValueError("hash length must be in range [1, 256]")


def validate_lsb_n(lsb_n: int):
    if not isinstance(lsb_n, int):
        raise TypeError("invalid lsb n")
    if lsb_n < 1 or lsb_n > 7:
        raise ValueError("lsb n must be in range [1, 7]")


def validate_non_empty_image(image: np.ndarray):
    """
    Checks to see if an object is a non-empty image.

    :raises TypeError: if the image is not a numpy image.
    :raises ValueError: if the image is empty or a 1-D array.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("object given as image is not a numpy image")
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("empty image")


def validate_image_array(image: np.ndarray):
    """
    Checks to see if an object is a non-empty 8-bit image.

    :raises TypeError: if the image is not a numpy image.
    :raises ValueError: if the image is empty or a 1-D array.
    :raises AssertionError: if the image is not an 8-bit image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("object given as image is not a numpy image")
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("empty image")
    assert image.dtype == np.uint8, "not an 8-bit image"


def validate_float_image_array(image: np.ndarray):
    """
    Checks to see if an object is a non-empty 32-bit float image.

    :raises TypeError: if the image is not a numpy image.
    :raises ValueError: if the image is empty or a 1-D array.
    :raises AssertionError: if the image is not an 32-bit float image.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("object given as image is not a numpy image")
    if image.ndim < 2 or image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("empty image")
    assert image.dtype == np.float32, "not an 32-bit float image"


def validate_image_for_preprocess(image: np.ndarray):
    """
    Checks to see if an object is a non-empty 8-bit image, grayscale or color, and square.

    :raises TypeError: if the image is not a numpy image.
    :raises ValueError: if the image is empty or a 1-D array.
    :raises AssertionError: if the image is neither grayscale nor color, or non-square, or not 8-bit.
    """
    validate_image_array(image)
    if image.ndim not in (2, 3):
        raise AssertionError("neither 2-channel grayscale nor 3-channel color image")
    if image.shape[0] != image.shape[1]:
        raise AssertionError("non-square image")


def validate_image(image: np.ndarray, block_size: int):
    """
    Checks to see if an object is a non-empty 8-bit image, grayscale, square,
    and its dimensions are divisible by block-size.

    :raises TypeError: if the image is not a numpy image.
    :raises ValueError: if the image is empty or a 1-D array.
    :raises AssertionError: if the image is not grayscale or non-square or \
        not 8-bit or its dimensions are not divisible by block-size.
    """
    validate_image_array(image)
    if image.ndim != 2:
        raise AssertionError("not a grayscale image")
    if image.shape[0] != image.shape[1]:
        raise AssertionError("non-square image")
    if image.shape[0] % block_size != 0 or image.shape[1] % block_size != 0:
        raise AssertionError("image with dimensions not divisible by block-size")


def validate_float_image(image: np.ndarray, block_size: int):
    """
    Checks to see if an object is a non-empty 32-bit float image, grayscale, square,
    and its dimensions are divisible by block-size.

    :raises TypeError: if the image is not a numpy image.
    :raises ValueError: if the image is empty or a 1-D array.
    :raises AssertionError: if the image is not grayscale or non-square or \
        not 32-bit float or its dimensions are not divisible by block-size.
    """
    validate_float_image_array(image)
    if image.ndim != 2:
        raise AssertionError("not a grayscale image")
    if image.shape[0] != image.shape[1]:
        raise AssertionError("non-square image")
    if image.shape[0] % block_size != 0 or image.shape[1] % block_size != 0:
        raise AssertionError("image with dimensions not divisible by block-size")


def validate_hashes(hashes: tuple[tuple[bytes, ...], ...], image_shape: tuple[int, ...],
                    block_size: int, hash_length: int):
    """
    Checks to see if an object is a valid non-empty 2-D tuple of hashes.

    :param hashesh: The matrix of hashes.
    :param image_shape: The shape of the image from which hashes are generated.
    :param block_size: The used block-size.
    :param hash_length: The common length of all hashes int bits.

    :raises TypeError: if the given 'hashes' has invalid type or contains objects of invalid type.
    :raises ValueError: if number of dimensions of 'hashes' is invalid or a hash's length is invalid.
    """
    validate_hash_length(hash_length)
    hash_length_bytes = hash_length_in_bytes(hash_length)
    # Check whole matrix
    if not isinstance(hashes, (tuple, list)):
        raise TypeError("'hashes' matrix is not a tuple/list")
    if len(hashes) == 0:
        raise ValueError("empty 'hashes' matrix")
    if len(hashes) != image_shape[0] / block_size:
        raise ValueError("invalid number of rows in 'hashes' matrix")
    # Check rows
    for i, row in enumerate(hashes):
        # Check row
        if not isinstance(row, (tuple, list)):
            raise TypeError(f"'hashes' row {i} is not a tuple/list")
        if len(row) == 0:
            raise ValueError(f"empty 'hashes' row {i}")
        if len(row) != image_shape[1] / block_size:
            raise ValueError(f"invalid number of hashes in 'hashes' row {i}")
        # Check hashes (cells)
        for j, _hash in enumerate(row):
            # Check hash (cell)
            if not isinstance(_hash, (bytes, bytearray)):
                raise TypeError(f"hash at ({i}, {j}) is not a bytes/bytearray instance")
            if len(_hash) != hash_length_bytes:
                raise ValueError(f"length of hash at ({i}, {j}) is invalid")
