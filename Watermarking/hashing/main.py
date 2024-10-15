import random
import os

import cv2 as cv
import numpy as np

from src.tasks import \
    read_image, \
    watermark, \
    watermark_is_authentic, \
    watermark_hash_comparison_table, \
    tamper_mask
from src.evaluation import psnr, ssim_similarity
from src.relative_rect import random_relative_rects
from src.corruption import Attack, \
    CutAttack, CopyAttack, MoveAttack, \
    RotateAttack, FlipAttack, \
    SaltAndPepperNoiseAttack, GaussianNoiseAttack, \
    GaussianFilterAttack, MedianFilterAttack, \
    HistogramEqualizationAttack, \
    JPEGEncodingAttack
from src.misc import cv_wait_for_esc_or_x, cv_destroy_windows_if_shown
from src.output import \
    save_watermarked_image, \
    save_watermarking_evaluation, \
    save_hash_comparison_table, \
    save_corrupt_and_detect_result, \
    save_multi_corrupt_and_detect_result


DEFAULT_IMAGE_PATH: str = "lenna.png"
DEFAULT_WATERMARKED_IMAGE_PATH: str = ".\\results\\lenna_watermarked.png"
DEFAULT_TAMPERED_IMAGE_PATH: str = ".\\results\\lenna_watermarked_tampered.png"
DEFAULT_RESULTS_DIR_PATH: str = ".\\results"

DEFAULT_ARNOLD_ITERATIONS: int = 8
DEFAULT_CELL_SIZE: int = 4
DEFAULT_BLOCK_SIZE: int = 8
DEFAULT_LSB_N: int = 1


def prepare_results_directory(results_directory: str) -> str:
    results_directory = os.path.realpath(results_directory)
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)
    return results_directory


def run_watermark() -> int:
    image_path = DEFAULT_IMAGE_PATH
    results_directory = DEFAULT_RESULTS_DIR_PATH
    block_size = DEFAULT_BLOCK_SIZE
    cell_size = DEFAULT_CELL_SIZE
    arnold_iterations = DEFAULT_ARNOLD_ITERATIONS
    lsb_n = DEFAULT_LSB_N
    # Watermark
    print(f"Watermarking image: '{image_path}'...")
    image = read_image(image_path)
    image, watermarked_image = watermark(image, block_size, cell_size, lsb_n, arnold_iterations)
    # Output
    prepare_results_directory(results_directory)
    file_path = save_watermarked_image(watermarked_image, results_directory, image_path)
    print(f"Saved watermarked image: '{file_path}'")
    _ssim = ssim_similarity(image, watermarked_image)
    _psnr, _mse = psnr(image, watermarked_image)
    file_path = save_watermarking_evaluation(_ssim, _mse, _psnr, results_directory, image_path)
    print(f"Saved evaluation result: '{file_path}'")
    # Display
    print(f"SSIM: {_ssim:.6f}")
    print(f"MSE: {_mse:.6f}")
    print(f"PSNR: {_psnr:.6f}")
    window_name = "Watermarked Image"
    cv.imshow(window_name, watermarked_image)
    cv_wait_for_esc_or_x(window_name)
    cv_destroy_windows_if_shown(window_name)


def run_authenticate_watermark() -> int:
    image_path = DEFAULT_WATERMARKED_IMAGE_PATH
    block_size = DEFAULT_BLOCK_SIZE
    cell_size = DEFAULT_CELL_SIZE
    arnold_iterations = DEFAULT_ARNOLD_ITERATIONS
    lsb_n = DEFAULT_LSB_N
    # Authenticate
    print(f"Authenticating watermarked image: '{image_path}'...")
    image = read_image(image_path)
    authentic_watermark = watermark_is_authentic(image, block_size, cell_size, lsb_n, arnold_iterations)
    # Display result
    print("Authentic." if authentic_watermark else "NOT Authentic.")


def run_compare_watermark_hashes() -> int:
    image_path = DEFAULT_WATERMARKED_IMAGE_PATH
    results_directory = DEFAULT_RESULTS_DIR_PATH
    block_size = DEFAULT_BLOCK_SIZE
    cell_size = DEFAULT_CELL_SIZE
    arnold_iterations = DEFAULT_ARNOLD_ITERATIONS
    lsb_n = DEFAULT_LSB_N
    # Generate comparisons
    print("Generating comparison table of generated-extracted hashes for watermaked image:")
    print(f"'{image_path}'...")
    image = read_image(image_path)
    table = watermark_hash_comparison_table(image, block_size, cell_size, lsb_n, arnold_iterations)
    # Result
    prepare_results_directory(results_directory)
    file_path = save_hash_comparison_table(table, results_directory, image_path)
    print(f"Saved hash-comparison table: '{file_path}'")


def run_detect_tampers() -> int:
    image_path = DEFAULT_TAMPERED_IMAGE_PATH
    block_size = DEFAULT_BLOCK_SIZE
    cell_size = DEFAULT_CELL_SIZE
    arnold_iterations = DEFAULT_ARNOLD_ITERATIONS
    lsb_n = DEFAULT_LSB_N
    # Detect tampered blocks
    image = read_image(image_path)
    mask = tamper_mask(image, block_size, cell_size, lsb_n, arnold_iterations)
    # Display result
    layer = cv.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
    blend = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    blend = cv.addWeighted(blend, 1., layer, 0.5, 0.)
    window_name_source = "Source Image"
    window_name_mask = "Tamper Mask"
    window_name_detection = "Tamper Detection Result"
    cv.imshow(window_name_source, image)
    cv.imshow(window_name_mask, mask)
    cv.imshow(window_name_detection, blend)
    window_names = window_name_source, window_name_mask, window_name_detection
    cv_wait_for_esc_or_x(window_names)
    cv_destroy_windows_if_shown(window_names)


def _run_corrupt_and_detect_attacks() -> tuple[Attack, ...]:
    cut_rect = random_relative_rects((0, 1), (0, 1), (0.1, 0.2), (0.1, 0.2), False, True, 1)[0]
    copy_rects = random_relative_rects((0, 1), (0, 1), (0.1, 0.2), (0.1, 0.2), False, True, 2)
    move_rects = random_relative_rects((0, 1), (0, 1), (0.1, 0.2), (0.1, 0.2), False, True, 2)
    return (
        CutAttack(cut_rect, random.randrange(256)),
        CopyAttack(copy_rects[0], copy_rects[1]),
        MoveAttack(move_rects[0], move_rects[1], random.randrange(256)),
        RotateAttack(90),
        RotateAttack(180),
        RotateAttack(270),
        FlipAttack(False),
        FlipAttack(True),
        SaltAndPepperNoiseAttack(0.12),
        GaussianNoiseAttack(0, 500),
        GaussianFilterAttack(11),
        MedianFilterAttack(11),
        HistogramEqualizationAttack(),
        JPEGEncodingAttack(20)
    )


def run_corrupt_and_detect() -> int:
    image_path = DEFAULT_IMAGE_PATH
    results_directory = DEFAULT_RESULTS_DIR_PATH
    block_size = DEFAULT_BLOCK_SIZE
    cell_size = DEFAULT_CELL_SIZE
    arnold_iterations = DEFAULT_ARNOLD_ITERATIONS
    lsb_n = DEFAULT_LSB_N
    # Preparation
    prepare_results_directory(results_directory)
    # Source image
    image = read_image(image_path)
    image, watermarked_image = watermark(image, block_size, cell_size, lsb_n, arnold_iterations)
    # Attacks
    for attack in _run_corrupt_and_detect_attacks():
        print(f"Applying attack: {attack.type}...", end="")
        tampered_image = attack.apply(watermarked_image)
        # Detect tampered blocks
        mask = tamper_mask(tampered_image, block_size, cell_size, lsb_n, arnold_iterations)
        layer = cv.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
        blend = cv.cvtColor(tampered_image, cv.COLOR_GRAY2BGR)
        blend = cv.addWeighted(blend, 1., layer, 0.5, 0.)
        print(f"\rAttack applied: {attack.type}.     ")
        # Output
        save_corrupt_and_detect_result(tampered_image, mask, blend, attack.type,
                                       results_directory, image_path)


def _run_corrupt_and_detect_multi_attack_attacks() -> tuple[Attack, ...]:
    move_rects = random_relative_rects((0, 1), (0, 1), (0.1, 0.2), (0.1, 0.2), False, True, 2)
    return (
        MoveAttack(move_rects[0], move_rects[1], random.randrange(256)),
        SaltAndPepperNoiseAttack(0.01),
    )


def run_corrupt_and_detect_multi_attack() -> int:
    image_path = DEFAULT_IMAGE_PATH
    results_directory = DEFAULT_RESULTS_DIR_PATH
    block_size = DEFAULT_BLOCK_SIZE
    cell_size = DEFAULT_CELL_SIZE
    arnold_iterations = DEFAULT_ARNOLD_ITERATIONS
    lsb_n = DEFAULT_LSB_N
    # Preparation
    prepare_results_directory(results_directory)
    # Source image
    image = read_image(image_path)
    image, watermarked_image = watermark(image, block_size, cell_size, lsb_n, arnold_iterations)
    # Attacks
    for attack in _run_corrupt_and_detect_multi_attack_attacks():
        print(f"Applying attack: {attack.type}...", end="")
        tampered_image = attack.apply(watermarked_image)
        watermarked_image = tampered_image
        print(f"\rAttack applied: {attack.type}.     ")
    # Detect tampered blocks
    mask = tamper_mask(tampered_image, block_size, cell_size, lsb_n, arnold_iterations)
    layer = cv.merge([np.zeros_like(mask), np.zeros_like(mask), mask])
    blend = cv.cvtColor(tampered_image, cv.COLOR_GRAY2BGR)
    blend = cv.addWeighted(blend, 1., layer, 0.5, 0.)
    # Display result
    window_name_source = "Source Image"
    window_name_tampered = "Tampered Image"
    window_name_mask = "Tamper Mask"
    window_name_detection = "Tamper Detection Result"
    cv.imshow(window_name_source, image)
    cv.imshow(window_name_tampered, watermarked_image)
    cv.imshow(window_name_mask, mask)
    cv.imshow(window_name_detection, blend)
    window_names = window_name_source, window_name_tampered, window_name_mask, window_name_detection
    cv_wait_for_esc_or_x(window_names)
    cv_destroy_windows_if_shown(window_names)
    # Output
    save_multi_corrupt_and_detect_result(tampered_image, mask, blend, results_directory, image_path)


if __name__ == "__main__":
    run_corrupt_and_detect()
