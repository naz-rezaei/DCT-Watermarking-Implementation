import os

import cv2 as cv
import numpy as np

from src.tasks import \
    preprocess_image, \
    arnold_transform, \
    inverse_arnold_transform, \
    blocks_dct_coefficients


DIR_PATH: str = ".\\visualize"
DIR_PATH_ARNOLD: str = ".\\visualize_arnold_effect_on_tamper"

ARNOLD_ITERATIONS: int = 2
BLOCK_SIZE: int = 4
HASH_LENGTH: int = 16
LSB_N: int = 1


def prepare_results_directory(results_directory: str) -> str:
    results_directory = os.path.realpath(results_directory)
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)
    return results_directory


def visualize_dct_coefficients(dct_coefficients: np.ndarray) -> np.ndarray:
    _min = float(dct_coefficients.min())
    _max = float(dct_coefficients.max())
    image = (dct_coefficients - _min) / (_max - _min) * 255.
    return np.clip(image, 0, 255).astype(np.uint8)


def run():
    dir_path = os.path.realpath(DIR_PATH)
    prepare_results_directory(dir_path)
    block_size = BLOCK_SIZE
    arnold_iterations = ARNOLD_ITERATIONS
    # Input
    image = cv.imread(os.path.join(dir_path, "_image.png"), flags=cv.IMREAD_GRAYSCALE)
    image = preprocess_image(image, block_size)
    cv.imwrite(os.path.join(dir_path, "wm_source.png"), image)
    # Zero LSB
    cv.imwrite(os.path.join(dir_path, "wm_zero_lsb.png"), image)
    # DCT coefficients
    dct_coeffs = blocks_dct_coefficients(image, block_size)
    dct_coeffs_image = visualize_dct_coefficients(dct_coeffs)
    cv.imwrite(os.path.join(dir_path, "wm_dct_coeffs.png"), dct_coeffs_image)
    # Hashes
    hash_image = cv.imread(os.path.join(dir_path, "_hash.png"), flags=cv.IMREAD_GRAYSCALE)
    cv.imwrite(os.path.join(dir_path, "wm_hashes.png"), hash_image)
    # Arnold
    scrambled_image = arnold_transform(image, arnold_iterations)
    cv.imwrite(os.path.join(dir_path, "wm_arnold.png"), scrambled_image)
    # Embed
    scrambled_watermarked_image = cv.add(scrambled_image, hash_image)
    cv.imwrite(os.path.join(dir_path, "wm_embed.png"), scrambled_watermarked_image)
    # Watermark
    watermarked_image = inverse_arnold_transform(scrambled_watermarked_image, arnold_iterations)
    cv.imwrite(os.path.join(dir_path, "wm_final.png"), watermarked_image)
    # -------------------------------------------------------------------------------------------------------
    # Tampered
    tamper = cv.imread(os.path.join(dir_path, "_tamper.png"), flags=cv.IMREAD_GRAYSCALE)
    tampered_watermarked_image = cv.add(watermarked_image, tamper)
    cv.imwrite(os.path.join(dir_path, "wm_tampered.png"), tampered_watermarked_image)
    # Tampered (Zero LSB)
    zero_lsb_of_tampered = cv.add(image, tamper)
    cv.imwrite(os.path.join(dir_path, "wm_tampered_zero_lsb.png"), zero_lsb_of_tampered)
    # Tampered (DCT coefficients)
    dct_coeffs = blocks_dct_coefficients(zero_lsb_of_tampered, block_size)
    tampered_dct_coeffs_image = visualize_dct_coefficients(dct_coeffs)
    cv.imwrite(os.path.join(dir_path, "wm_tampered_dct_coeffs.png"), tampered_dct_coeffs_image)
    # Tampered (Hashes)
    tampered_hash_image = cv.imread(os.path.join(dir_path, "_tampered_hash.png"), flags=cv.IMREAD_GRAYSCALE)
    cv.imwrite(os.path.join(dir_path, "wm_tampered_hashes.png"), tampered_hash_image)
    # Tampered (Arnold)
    tampered_scrambled_image = arnold_transform(tampered_watermarked_image, arnold_iterations)
    cv.imwrite(os.path.join(dir_path, "wm_tampered_arnold.png"), tampered_scrambled_image)
    # Tampered (Extract hashes)
    tamper_arnold = arnold_transform(tamper, arnold_iterations)
    tampered_extracted_hash = np.maximum(hash_image, tamper_arnold / 2)
    cv.imwrite(os.path.join(dir_path, "wm_tampered_extracted_hashes.png"), tampered_extracted_hash)


def run_visualize_arnold_effect_on_tamper():
    dir_path = os.path.realpath(DIR_PATH_ARNOLD)
    prepare_results_directory(dir_path)
    image = np.zeros((512, 512), dtype=np.uint8)
    image[252:260, 252:260] = np.full((8, 8), 255, dtype=np.uint8)
    cv.imwrite(os.path.join(dir_path, f"{0:02d}.png"), image)
    for arnold_iterations in range(1, 21):
        inverse_arnold = inverse_arnold_transform(image, arnold_iterations)
        result_image = cv.addWeighted(image, 0.5, inverse_arnold, 1.0, 0.0)
        cv.imwrite(os.path.join(dir_path, f"{arnold_iterations:02d}.png"), result_image)


def arn():
    path = r"C:\Users\Hamid\Desktop\nrn\graphics"
    n = 1
    img = cv.imread(os.path.join(path, "block.png"))
    res_img = arnold_transform(img, n)
    cv.imwrite(os.path.join(path, "block_arnold.png"), res_img)
    dct = cv.dct(cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32))
    _max = np.max(dct)
    _min = np.min(dct)
    dct = (((dct - _min) / (_max - _min)) * 255).astype(np.uint8)
    cv.imwrite(os.path.join(path, "dct.png"), dct)
    img = cv.imread(os.path.join(path, "hash.png"))
    res_img = inverse_arnold_transform(img, n)
    cv.imwrite(os.path.join(path, "hash_inv_arnold.png"), res_img)


if __name__ == "__main__":
    #run_visualize_arnold_effect_on_tamper()
    arn()
