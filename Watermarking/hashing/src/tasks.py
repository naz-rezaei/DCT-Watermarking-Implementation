from hashlib import sha256
import numpy as np
import cv2 as cv

from src.misc import hash_length_in_bytes, bytes_to_hex, calculate_hash_length
from src.validation import \
    validate_block_and_cell_size, \
    validate_hash_length, \
    validate_lsb_n, \
    validate_arnold_iterations, \
    validate_non_empty_image, \
    validate_image_array, \
    validate_image_for_preprocess, \
    validate_image, \
    validate_float_image, \
    validate_hashes


def preprocess_image(image: np.ndarray, block_size: int) -> np.ndarray:
    """Resizes the image so that its dimensions are divisible by block-size."""
    validate_image_for_preprocess(image)
    validate_block_and_cell_size(block_size, 1)
    # Resize
    dim_converter = lambda v: int(v) if int(v) % block_size == 0 \
                              else (int(v / block_size) + 1) * block_size 
    w, h = dim_converter(image.shape[1]), dim_converter(image.shape[0])
    if w != image.shape[1] or h != image.shape[0]:
        return cv.resize(image, dsize=(w, h), interpolation=cv.INTER_CUBIC)
    return image.copy()


def set_lsbs_to_zero(image: np.ndarray, lsb_n: int = 1) -> np.ndarray:
    """Sets a desired number of least-significant bits in all pixels of an image to zero."""
    validate_image_array(image)
    assert isinstance(lsb_n, int) and 0 <= lsb_n <= 8, "invalid/unsupported number of lsb bits"
    return np.left_shift(np.right_shift(image, lsb_n), lsb_n)


def arnold_transform(image: np.ndarray, iterations: int):
    """
    Generates the Arnold transform of an image.

    :param image: The source image.
    :param iterations: The number of iterations of Arnold transform.
    """
    validate_non_empty_image(image)
    validate_arnold_iterations(iterations)
    if iterations == 0:
        return image.copy()
    is_gray = image.ndim == 2
    tm = np.array([[1, 1], [1, 2]])
    result = np.zeros_like(image)
    shape = image.shape[0:2]
    indexes = (tm @ np.indices(shape).reshape(2, -1)) % (np.array(shape)[:, None])
    if is_gray:
        for _ in range(iterations):
            result[indexes[0], indexes[1]] = image.reshape(-1)
            image = result.copy()
    else:
        for _ in range(iterations):
            result[indexes[0], indexes[1]] = image.reshape((shape[0] * shape[1], -1))
            image = result.copy()
    return result


def inverse_arnold_transform(scrambled_image: np.ndarray, iterations: int):
    """
    Generates the inverse Arnold transform of an scrambled image using Arnold transform.

    :param image: The source scrambled image.
    :param iterations: The number of iterations of inverse Arnold transform.
    """
    validate_non_empty_image(scrambled_image)
    validate_arnold_iterations(iterations)
    if iterations == 0:
        return scrambled_image.copy()
    is_gray = scrambled_image.ndim == 2
    tm = np.array([[2, -1], [-1, 1]])
    result = np.zeros_like(scrambled_image)
    shape = scrambled_image.shape[0:2]
    indexes = (tm @ np.indices(shape).reshape(2, -1)) % (np.array(shape)[:, None])
    if is_gray:
        for _ in range(iterations):
            result[indexes[0], indexes[1]] = scrambled_image.reshape(-1)
            scrambled_image = result.copy()
    else:
        for _ in range(iterations):
            result[indexes[0], indexes[1]] = scrambled_image.reshape((shape[0] * shape[1], -1))
            scrambled_image = result.copy()
    return result


def blocks_dct_coefficients(image: np.ndarray, block_size: int, cell_size: int) -> np.ndarray:
    """
    Calculates DCT coefficients of pixels of all blocks' cells in the source image.

    :param image: The source image.
    :param block_size: The size that is used to block the source image.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_image(image, block_size)
    blocks_dct_coeffs = np.zeros_like(image, dtype=np.float32)
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            patch = image[i:i + cell_size, j:j + cell_size].astype(np.float32)
            blocks_dct_coeffs[i:i + cell_size, j:j + cell_size] = cv.dct(patch)
    return blocks_dct_coeffs


def hash_blocks(blocks_dct_coeffs: np.ndarray, block_size: int, cell_size: int,
                lsb_n: int) -> tuple[tuple[bytes, ...], ...]:
    """
    Hashes all blocks of a DCT coefficiencts matrix using SHA256 algorithm.

    :param blocks_dct_coeffs: The source DCT coefficiencts matrix.
    :return: The 2-D matrix of produced hashes.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_float_image(blocks_dct_coeffs, block_size)
    hash_length_bytes = hash_length_in_bytes(hash_length)
    hashes = []
    for i in range(0, blocks_dct_coeffs.shape[0], block_size):
        row = []
        for j in range(0, blocks_dct_coeffs.shape[1], block_size):
            _hash = sha256(blocks_dct_coeffs[i:i + cell_size, j:j + cell_size].tobytes()).digest()
            row.append(_hash[:hash_length_bytes])
        hashes.append(tuple(row))
    return tuple(hashes)


def unpack_hash(_hash: bytes | bytearray, hash_length: int, lsb_n: int) -> bytearray:
    """
    Unpacks a hash to a sequence of bytes in which each byte stores 'lsb_n' bits of the original hash.

    :param _hash: The original hash.
    :param hash_length: The length of hash in bits.
    :param lsb_n: The number of bits that are allocated for hash storage in each pixel.
    """
    # A unit is a bunch of consecutive bits that are going to be embedded in ONE pixel.
    units = bytearray()
    unit = 0x00
    unit_size = 0
    total_size = 0  # This is used to store how many bits are written to 'units'.
    for b in _hash:
        for i in range(8):
            # Acquire the bit
            bit = (b >> i) & 0x01
            # Write the bit to the unit
            unit |= bit << unit_size
            unit_size += 1
            total_size += 1
            # Check if the unit's capacity is reached
            if unit_size >= lsb_n:
                # Finish the unit up
                units.append(unit)
                unit = 0x00
                unit_size = 0
            # Break if all hash-bits are unpacked
            if total_size >= hash_length:
                break
        # Break if all hash-bits are unpacked
        if total_size >= hash_length:
            break
    # Finish the last non-empty unit up
    if unit_size > 0:
        units.append(unit)
    return units


def embed_cell_hash(image: np.ndarray, i: int, j: int, cell_size: int,
                    _hash: bytes | bytearray, hash_length: int, lsb_n: int):
    """
    Embeds a cell's hash on its pixels.

    :param image: The soure image (array).
    :param i: The row index on the image from which the cell expands.
    :param j: The column index on the image from which the cell expands.
    :param cell_size: The cell-size.
    :param _hash: The cell's hash.
    :param hash_length: Length of the hash in bits.
    :param lsb_n: Number of lsbs.
    """
    content_mask = (0xFF >> lsb_n) << lsb_n
    cell_hash_bytes = unpack_hash(_hash, hash_length, lsb_n)
    for cell_hash_byte_index, cell_hash_byte in enumerate(cell_hash_bytes):
        cell_i = i + cell_hash_byte_index // cell_size
        cell_j = j + cell_hash_byte_index % cell_size
        image[cell_i, cell_j] = (int(image[cell_i, cell_j]) & content_mask) | cell_hash_byte

    
def embed_hashes(image: np.ndarray, hashes: tuple[tuple[bytes, ...], ...],
                 block_size: int, cell_size: int, lsb_n: int = 1) -> np.ndarray:
    """
    Embeds blocks hashes on their pixels.

    :param image: The soure image (array).
    :param hashes: Hashes of the image's blocks.
    :param block_size: The block-size.
    :param hash_length: Length of the hash in bits.
    :param lsb_n: Number of lsbs.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_image(image, block_size)
    validate_hashes(hashes, image.shape, block_size, hash_length)
    image = image.copy()
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            embed_cell_hash(image, i, j, cell_size,
                            hashes[i // block_size][j // block_size], hash_length, lsb_n)
    return image


def read_cell_hash(image: np.ndarray, i: int, j: int, cell_size: int,
                   hash_length: int, lsb_n: int) -> bytes:
    """
    Reads a block's hash from its pixels.

    :param image: The soure image (array).
    :param i: The row index on the image from which the cell expands.
    :param j: The column index on the image from which the cell expands.
    :param cell_size: The cell-size.
    :param _hash: The block's hash.
    :param hash_length: Length of the hash in bits.
    :param lsb_n: Number of lsbs.
    """
    lsbs_mask = ~((0xFF >> lsb_n) << lsb_n) & 0xFF  # The mask for reading lsbs from each pixel
    _hash = bytearray()
    # Initialize
    remaining_bits_to_read = hash_length
    written_bits_to_hash = 0
    buffer = 0x0000
    buffer_size = 0
    # Read pixel by pixel
    for pixel_index in range(cell_size * cell_size):
        # Break if all of the hash's bits are read
        if remaining_bits_to_read <= 0:
            break
        # The pixel's coordinates
        cell_i = i + pixel_index // cell_size
        cell_j = j + pixel_index % cell_size
        # Read the pixel's lsbs into the buffer
        pixel_lsbs = int(image[cell_i, cell_j]) & lsbs_mask
        buffer |= pixel_lsbs << buffer_size
        buffer_size += lsb_n
        remaining_bits_to_read -= lsb_n
        # Extract whole bytes from the buffer (read bits)
        while buffer_size >= 8:
            _hash.append(buffer & 0xFF)
            buffer >>= 8
            written_bits_to_hash += 8
            buffer_size -= 8
    # Write remaining bits from the buffer
    if written_bits_to_hash < hash_length:
        assert buffer_size >= hash_length - written_bits_to_hash, "buffer underflow"
        buffer_size = hash_length - written_bits_to_hash  # Actual buffer-size at the end
        # The mask for reading remaining bits from the buffer
        buffer_mask = ~((0xFF >> buffer_size) << buffer_size) & 0xFF
        _hash.append(buffer & buffer_mask)
    return bytes(_hash)


def extract_hashes(image: np.ndarray, block_size: int, cell_size: int,
                   lsb_n: int) -> tuple[tuple[bytes, ...], ...]:
    """
    Extracts hashes from blocks of an image.

    :param image: The image.
    :param block_size: The block-size.
    :param cell_size: The cell-size.
    :param lsb_n: The number of least-siginificant bits on each pixel of the image \
        allocated for hash storage.
    :return: The blocks' hashes.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_image(image, block_size)
    hashes = []
    for i in range(0, image.shape[0], block_size):
        row = []
        for j in range(0, image.shape[1], block_size):
            _hash = read_cell_hash(image, i, j, cell_size, hash_length, lsb_n)
            row.append(_hash)
        hashes.append(tuple(row))
    return tuple(hashes)


def apply_arnold_to_blocks(image: np.ndarray, block_size: int, cell_size: int, arnold_iterations: int):
    """
    Applies Arnold transform independently to all blocks of an image.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_arnold_iterations(arnold_iterations)
    validate_image(image, block_size)
    image = image.copy()
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            image[i:i + cell_size, j:j + cell_size] \
                = arnold_transform(image[i:i + cell_size, j:j + cell_size], arnold_iterations)
    return image


def apply_inverse_arnold_to_blocks(image: np.ndarray, block_size: int, cell_size: int, arnold_iterations: int):
    """
    Applies inverse Arnold transform independently to all blocks of an image.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_arnold_iterations(arnold_iterations)
    validate_image(image, block_size)
    image = image.copy()
    for i in range(0, image.shape[0], block_size):
        for j in range(0, image.shape[1], block_size):
            image[i:i + cell_size, j:j + cell_size] \
                = inverse_arnold_transform(image[i:i + cell_size, j:j + cell_size], arnold_iterations)
    return image


def read_image(image_path: str) -> np.ndarray:
    """
    Reads the image provided by a given path using OpenCV's ``imread`` as an 8-bit image.

    :param image_path: Path to the image file.
    :raises RuntimeError: if fails to load the image.
    """
    if not isinstance(image_path, str):
        raise TypeError(f"string image path required")
    image = cv.imread(image_path, flags=cv.IMREAD_GRAYSCALE)
    try:
        validate_image_array(image)
    except (TypeError, ValueError) as ex:
        raise RuntimeError(f"failed to load image '{image_path}': {ex}")
    return image


def watermark(image: np.ndarray, block_size: int, cell_size: int, lsb_n: int,
              arnold_iterations: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Watermarks an image.

    Note: The hash should fit in the block leaving the least possible unused bits.
    Formally, the following inequality should hold:

    ``block-size x block-size x (lsb-n - 1) < hash-length <= block-size x block-size x lsb-n``

    :param image: The source image.
    :param block_size: The block-size.
    :param cell_size: The cell-size.
    :param lsb_n: The number of least-siginificant bits on each pixel of the image \
        allocated for hash storage.
    :param arnold_iterations: The number of iteration used for Arnold transform.
    :return: The pre-processed image and the watermarked image.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_arnold_iterations(arnold_iterations)
    validate_image_for_preprocess(image)
    # Preprocess
    preprocessed_image = preprocess_image(image, block_size)
    validate_image(preprocessed_image, block_size)
    # Generate hashes
    image = set_lsbs_to_zero(preprocessed_image, lsb_n)
    dct_coeffs = blocks_dct_coefficients(image, block_size, cell_size)
    hashes = hash_blocks(dct_coeffs, block_size, cell_size, lsb_n)
    # Embed hashes
    scrambled_image = apply_arnold_to_blocks(image, block_size, cell_size, arnold_iterations)
    scrambled_watermarked_image = embed_hashes(scrambled_image, hashes, block_size, cell_size, lsb_n)
    return (
        preprocessed_image,
        apply_inverse_arnold_to_blocks(scrambled_watermarked_image, block_size, cell_size, arnold_iterations)
    )


def watermark_is_authentic(watermarked_image: np.ndarray, block_size: int, cell_size: int,
                           lsb_n: int, arnold_iterations: int) -> bool:
    """
    Verifies the authenticity of an image's watermark.

    :param watermarked_image: The source watermarked image.
    :param block_size: The block-size.
    :param cell_size: The cell-size.
    :param lsb_n: The number of least-siginificant bits on each pixel of the image \
        allocated for hash storage.
    :param arnold_iterations: The number of iteration used for Arnold transform.
    :return: The blocks' hashes.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_arnold_iterations(arnold_iterations)
    validate_image(watermarked_image, block_size)
    # Extract hashes
    scrambled_watermarked_image = apply_arnold_to_blocks(watermarked_image, block_size, cell_size, arnold_iterations)
    extracted_hashes = extract_hashes(scrambled_watermarked_image, block_size, cell_size, lsb_n)
    # Generate hashes
    watermarked_image = set_lsbs_to_zero(watermarked_image, lsb_n)
    dct_coeffs = blocks_dct_coefficients(watermarked_image, block_size, cell_size)
    generated_hashes = hash_blocks(dct_coeffs, block_size, cell_size, lsb_n)
    # Compare extracted hashes to generated hashes
    authentic = True
    if len(extracted_hashes) != len(generated_hashes):
        raise ValueError("mismatching block-size")
    for i, extracted_row in enumerate(extracted_hashes):
        generated_row = generated_hashes[i]
        if len(extracted_row) != len(generated_row):
            raise ValueError("mismatching block-size")
        for j, extracted_hash in enumerate(extracted_row):
            generated_hash = generated_row[j]
            if extracted_hash != generated_hash:
                authentic = False
                break
        if not authentic:
            break
    return authentic


def watermark_hash_comparison_table(watermarked_image: np.ndarray, block_size: int, cell_size: int,
                                    lsb_n: int, arnold_iterations: int) -> tuple[tuple[str, ...], ...]:
    """
    Produces the table of comparison of embedded hashes and comparison hashes.

    :param watermarked_image: The source watermarked image.
    :param block_size: The block-size.
    :param cell_size: The cell-size.
    :param lsb_n: The number of least-siginificant bits on each pixel of the image \
        allocated for hash storage.
    :param arnold_iterations: The number of iteration used for Arnold transform.
    :return: The table.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_arnold_iterations(arnold_iterations)
    validate_image(watermarked_image, block_size)
    # Extract hashes
    scrambled_watermarked_image = apply_arnold_to_blocks(watermarked_image, block_size, cell_size, arnold_iterations)
    extracted_hashes = extract_hashes(scrambled_watermarked_image, block_size, cell_size, lsb_n)
    # Generate hashes
    watermarked_image = set_lsbs_to_zero(watermarked_image, lsb_n)
    dct_coeffs = blocks_dct_coefficients(watermarked_image, block_size, cell_size)
    generated_hashes = hash_blocks(dct_coeffs, block_size, cell_size, lsb_n)
    # Compare extracted hashes to generated hashes
    table = []
    if len(extracted_hashes) != len(generated_hashes):
        raise ValueError("mismatching block-size")
    for i, extracted_row in enumerate(extracted_hashes):
        generated_row = generated_hashes[i]
        if len(extracted_row) != len(generated_row):
            raise ValueError("mismatching block-size")
        table_row = []
        for j, extracted_hash in enumerate(extracted_row):
            generated_hash = generated_row[j]
            table_row.append(f"{bytes_to_hex(generated_hash)}-{bytes_to_hex(extracted_hash)}")
        table.append(tuple(table_row))
    return tuple(table)


def tamper_mask(watermarked_image: np.ndarray, block_size: int, cell_size: int,
                lsb_n: int, arnold_iterations: int) -> np.ndarray:
    """
    Detects the tampered blocks of a watermarked image.

    :param image: The source watermarked image.
    :param block_size: The block-size.
    :param cell_size: The cell-size.
    :param lsb_n: The number of least-siginificant bits on each pixel of the image \
        allocated for hash storage.
    :param arnold_iterations: The number of iteration used for Arnold transform.
    :return: The tamper mask.
    """
    validate_block_and_cell_size(block_size, cell_size)
    validate_lsb_n(lsb_n)
    hash_length = calculate_hash_length(cell_size, lsb_n)
    validate_hash_length(hash_length)
    validate_arnold_iterations(arnold_iterations)
    validate_image(watermarked_image, block_size)
    # Generate hashes
    zero_lsbs_watermarked_image = set_lsbs_to_zero(watermarked_image, lsb_n)
    dct_coeffs = blocks_dct_coefficients(zero_lsbs_watermarked_image, block_size, cell_size)
    generated_hashes = hash_blocks(dct_coeffs, block_size, cell_size, lsb_n)
    # Extract hashes
    scrambled_watermarked_image = apply_arnold_to_blocks(watermarked_image, block_size, cell_size, arnold_iterations)
    extracted_hashes = extract_hashes(scrambled_watermarked_image, block_size, cell_size, lsb_n)
    # Compare extracted hashes to generated hashes and generate the tamper mask
    mask = np.zeros(watermarked_image.shape, np.uint8)
    if len(extracted_hashes) != len(generated_hashes):
        raise ValueError("mismatching block-size")
    for i, extracted_row in enumerate(extracted_hashes):
        generated_row = generated_hashes[i]
        if len(extracted_row) != len(generated_row):
            raise ValueError("mismatching block-size")
        for j, extracted_hash in enumerate(extracted_row):
            generated_hash = generated_row[j]
            if extracted_hash != generated_hash:
                oi = i * block_size
                oj = j * block_size
                mask[oi:oi + block_size, oj:oj + block_size] \
                    = np.full((block_size, block_size), 255, dtype=np.uint8)
    return mask
