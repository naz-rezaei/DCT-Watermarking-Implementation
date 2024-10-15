import os
import numpy as np
import cv2 as cv


def save_watermarked_image(watermarked_image: np.ndarray,
                           results_directory: str, source_image_file_name: str) -> str:
    source_image_name = os.path.splitext(os.path.basename(source_image_file_name))[0]
    dest_image_name = f"{source_image_name}_watermarked.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, watermarked_image)
    return dest_image_path


def save_watermarking_evaluation(_ssim: float, _mse: float, _psnr: float,
                                 results_directory: str, source_image_file_name: str) -> str:
    source_image_name = os.path.splitext(os.path.basename(source_image_file_name))[0]
    dest_file_name = f"{source_image_name}_watermarking_eval.txt"
    dest_file_path = os.path.join(results_directory, dest_file_name) if results_directory \
        else dest_file_name
    with open(dest_file_path, "w") as _file:
        _file.write(f"SSIM: {_ssim:.6f}\n")
        _file.write(f"MSE: {_mse:.6f}\n")
        _file.write(f"PSNR: {_psnr:.6f}\n")
    return dest_file_path


def save_hash_comparison_table(table: tuple[tuple[str, ...], ...],
                               results_directory: str, source_image_file_name: str) -> str:
    source_image_name = os.path.splitext(os.path.basename(source_image_file_name))[0]
    dest_file_name = f"{source_image_name}_hash_comparison_gen_emb.csv"
    dest_file_path = os.path.join(results_directory, dest_file_name) if results_directory \
        else dest_file_name
    with open(dest_file_path, "w") as _file:
        for row in table:
            for i, field in enumerate(row):
                if i > 0:
                    _file.write(",")
                _file.write(field)
            _file.write("\n")
    return dest_file_path


def save_corrupt_and_detect_result(tampered_image: np.ndarray, tamper_mask: np.ndarray, blended_image: np.ndarray,
                                   attack_name: str, results_directory: str, source_image_file_name: str):
    source_image_name = os.path.splitext(os.path.basename(source_image_file_name))[0]
    attack_name = attack_name.replace(" ", "_").lower()
    # Tampered
    dest_image_name = f"{source_image_name}_wmxt_{attack_name}.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, tampered_image)
    # Tamper mask
    dest_image_name = f"{source_image_name}_wmxt_{attack_name}_tamper_mask.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, tamper_mask)
    # Tamper mask blended with the image
    dest_image_name = f"{source_image_name}_wmxt_{attack_name}_tamper_mask_blended.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, blended_image)


def save_multi_corrupt_and_detect_result(tampered_image: np.ndarray, tamper_mask: np.ndarray, blended_image: np.ndarray,
                                         results_directory: str, source_image_file_name: str):
    source_image_name = os.path.splitext(os.path.basename(source_image_file_name))[0]
    # Tampered
    dest_image_name = f"{source_image_name}_wmxtmulti.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, tampered_image)
    # Tamper mask
    dest_image_name = f"{source_image_name}_wmxtmulti_tamper_mask.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, tamper_mask)
    # Tamper mask blended with the image
    dest_image_name = f"{source_image_name}_wmxtmulti_tamper_mask_blended.png"
    dest_image_path = os.path.join(results_directory, dest_image_name) if results_directory \
        else dest_image_name
    cv.imwrite(dest_image_path, blended_image)
