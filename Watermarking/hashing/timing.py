from time import perf_counter_ns
from typing import TextIO
from math import inf
import os

from src.tasks import \
    read_image, \
    preprocess_image, \
    set_lsbs_to_zero, \
    blocks_dct_coefficients, \
    hash_blocks, \
    apply_arnold_to_blocks, \
    embed_hashes, \
    apply_inverse_arnold_to_blocks, \
    watermark_is_authentic


IMAGE_PATHS: str = [
    "lenna_128.png",
    "lenna_256.png",
    "lenna_384.png",
    "lenna_512.png",
    "lenna_640.png",
    "lenna_768.png",
    "lenna_896.png",
    "lenna_1024.png",
]
RESULTS_DIR_PATH: str = ".\\timing"
ITERATIONS: int = 10

ARNOLD_ITERATIONS: int = 5
BLOCK_SIZE: int = 8
CELL_SIZE: int = 4
LSB_N: int = 1


def prepare_results_directory(results_directory: str) -> str:
    results_directory = os.path.realpath(results_directory)
    if not os.path.isdir(results_directory):
        os.makedirs(results_directory)
    return results_directory


def write_times(csv_file: TextIO, row_title: str,
                time_zero_lsb: int, time_dct_coeffs: int, time_hashing: int,
                time_arnold: int, time_embedding: int, time_inverse_arnold: int,
                watermarking_times_sum: int, watermarking_duration: int, time_authentication):
    csv_file.write(f"{row_title}," +
                   f"{round(time_zero_lsb / 1000000, 3)},{round(time_dct_coeffs / 1000000, 3)}," +
                   f"{round(time_hashing / 1000000, 3)},{round(time_arnold / 1000000, 3)}," +
                   f"{round(time_embedding / 1000000, 3)},{round(time_inverse_arnold / 1000000, 3)}," +
                   f"{round(watermarking_times_sum / 1000000, 3)},{round(watermarking_duration / 1000000, 3)}," +
                   f"{round(time_authentication / 1000000, 3)}\n")
    

def run_single(image_path: str, results_directory: str,
               block_size: int, cell_size: int, arnold_iterations: int, lsb_n: int) -> int:
    # Initialize
    prepare_results_directory(results_directory)
    image = read_image(image_path)
    preprocessed_image = preprocess_image(image, block_size)
    file_name = "timing_" + os.path.basename(image_path) + ".csv"
    with open(os.path.join(results_directory, file_name), "w", encoding="utf-8") as csv_file:
        csv_file.write("Iteration,Set LSB to Zero (ms),DCT Calculation (ms),Hashing (ms),Arnold (ms)," +
                       "Embedding (ms),Inverse Arnold (ms),Watermarking Times' Sum (ms),Watermarking Duration (ms)," +
                       "Authentication (ms)\n")
        min_times = inf, inf, inf, inf, inf, inf, inf, inf, inf
        max_times = -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf
        sum_times = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for iteration in range(ITERATIONS):
            print(f"Iteration {iteration + 1}")
            checkpoint = start = perf_counter_ns()
            #
            image = set_lsbs_to_zero(preprocessed_image, lsb_n)
            #
            now = perf_counter_ns()
            time_zero_lsb = now - checkpoint
            checkpoint = now
            #
            dct_coeffs = blocks_dct_coefficients(image, block_size, cell_size)
            #
            now = perf_counter_ns()
            time_dct_coeffs = now - checkpoint
            checkpoint = now
            #
            hashes = hash_blocks(dct_coeffs, block_size, cell_size, lsb_n)
            #
            now = perf_counter_ns()
            time_hashing = now - checkpoint
            checkpoint = now
            #
            scrambled_image = apply_arnold_to_blocks(image, block_size, cell_size, arnold_iterations)
            #
            now = perf_counter_ns()
            time_arnold = now - checkpoint
            checkpoint = now
            #
            scrambled_watermarked_image = embed_hashes(scrambled_image, hashes, block_size, cell_size, lsb_n)
            #
            now = perf_counter_ns()
            time_embedding = now - checkpoint
            checkpoint = now
            #
            watermarked_image = apply_inverse_arnold_to_blocks(scrambled_watermarked_image,
                                                               block_size, cell_size, arnold_iterations)
            #
            now = perf_counter_ns()
            time_inverse_arnold = now - checkpoint
            checkpoint = now
            watermarking_duration = now - start
            #
            watermark_is_authentic(watermarked_image, block_size, cell_size, lsb_n, arnold_iterations)
            #
            now = perf_counter_ns()
            time_authentication = now - checkpoint
            #
            watermarking_times_sum = time_zero_lsb + time_dct_coeffs + time_hashing \
                 + time_arnold + time_embedding + time_inverse_arnold
            min_times = (
                min(min_times[0], time_zero_lsb),
                min(min_times[1], time_dct_coeffs),
                min(min_times[2], time_hashing),
                min(min_times[3], time_arnold),
                min(min_times[4], time_embedding),
                min(min_times[5], time_inverse_arnold),
                min(min_times[6], watermarking_times_sum),
                min(min_times[7], watermarking_duration),
                min(min_times[8], time_authentication)
            )
            max_times = (
                max(max_times[0], time_zero_lsb),
                max(max_times[1], time_dct_coeffs),
                max(max_times[2], time_hashing),
                max(max_times[3], time_arnold),
                max(max_times[4], time_embedding),
                max(max_times[5], time_inverse_arnold),
                max(max_times[6], watermarking_times_sum),
                max(max_times[7], watermarking_duration),
                max(max_times[8], time_authentication)
            )
            sum_times = (
                sum_times[0] + time_zero_lsb,
                sum_times[1] + time_dct_coeffs,
                sum_times[2] + time_hashing,
                sum_times[3] + time_arnold,
                sum_times[4] + time_embedding,
                sum_times[5] + time_inverse_arnold,
                sum_times[6] + watermarking_times_sum,
                sum_times[7] + watermarking_duration,
                sum_times[8] + time_authentication
            )
            #
            write_times(csv_file, str(iteration + 1),
                        time_zero_lsb, time_dct_coeffs, time_hashing, time_arnold,
                        time_embedding, time_inverse_arnold, watermarking_times_sum, watermarking_duration,
                        time_authentication)
        #
        write_times(csv_file, "minimum",
                    min_times[0], min_times[1], min_times[2], min_times[3],
                    min_times[4], min_times[5], min_times[6], min_times[7], min_times[8])
        write_times(csv_file, "maximum",
                    max_times[0], max_times[1], max_times[2], max_times[3],
                    max_times[4], max_times[5], max_times[6], max_times[7], max_times[8])
        write_times(csv_file, "mean",
                    sum_times[0] / ITERATIONS, sum_times[1] / ITERATIONS,
                    sum_times[2] / ITERATIONS, sum_times[3] / ITERATIONS,
                    sum_times[4] / ITERATIONS, sum_times[5] / ITERATIONS,
                    sum_times[6] / ITERATIONS, sum_times[7] / ITERATIONS,
                    sum_times[8] / ITERATIONS)


def run() -> int:
    results_directory = RESULTS_DIR_PATH
    block_size = BLOCK_SIZE
    cell_size = CELL_SIZE
    arnold_iterations = ARNOLD_ITERATIONS
    lsb_n = LSB_N
    for image_path in IMAGE_PATHS:
        print("\n" + image_path)
        run_single(image_path, results_directory, block_size, cell_size, arnold_iterations, lsb_n)


if __name__ == "__main__":
    run()
