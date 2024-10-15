from argparse import ArgumentParser


def read_image_from_args():
    parser = ArgumentParser()
    parser.add_argument("image", help="Path to the image file.")
    args = parser.parse_args()
    return args.image
