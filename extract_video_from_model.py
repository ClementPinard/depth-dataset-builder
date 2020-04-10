from colmap_util import read_model as rm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd

parser = ArgumentParser(description='create a new colmap model with only the frames of selected video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_list', metavar='PATH',
                    help='path to list with relative path to images', type=Path)
parser.add_argument('--input_model', metavar='DIR', type=Path)
parser.add_argument('--output_model', metavar='DIR', default=None, type=Path)
parser.add_argument('--output_format', choices=['.txt', '.bin'], default='.txt')
parser.add_argument('--metadata_path', metavar="CSV", type=Path)


def extract_video(input, output, video_metadata_path, output_format='.bin'):
    cameras = rm.read_cameras_binary(input / "cameras.bin")
    images = rm.read_images_binary(input / "images.bin")
    images_per_name = {}
    video_metadata = pd.read_csv(video_metadata_path)
    image_names = video_metadata["image_path"].values
    for id, image in images.items():
        if image.name in image_names:
            image._replace(xys=[])
            image._replace(point3D_ids=[])
            images_per_name[image.name] = image
    camera_ids = video_metadata["camera_id"].unique()
    output_cameras = {cid: cameras[cid] for cid in camera_ids if cid in cameras.keys()}

    rm.write_model(output_cameras, images_per_name, {}, output, output_format)

    return len(images_per_name) > 1


def main():
    args = parser.parse_args()
    extract_video(args.input_model, args.output_model, args.metadata_path, args.output_format)
    return


if __name__ == '__main__':
    main()
