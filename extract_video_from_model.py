from colmap_util import read_model as rm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd

parser = ArgumentParser(description='create a new colmap model with only the frames of selected video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_model', metavar='DIR', type=Path, required=True,
                    help='folder where the cameras.bin and images.bin are located')
parser.add_argument('--output_model', metavar='DIR', type=Path, required=True,
                    help='Output folder where the modified COLMAP model will be saved')
parser.add_argument('--output_format', choices=['.txt', '.bin'], default='.txt')
parser.add_argument('--metadata_path', metavar="CSV", type=Path, required=True,
                    help='Path to metadata CSV file of the desired video. '
                    'Usually in /pictures/Videos/<size>/<video_name>/metadata.csv')


def extract_video(input, output, video_metadata_path, output_format='.bin'):
    cameras = rm.read_cameras_binary(input / "cameras.bin")
    images = rm.read_images_binary(input / "images.bin")
    images_per_name = {}
    video_metadata = pd.read_csv(video_metadata_path)
    image_names = video_metadata["image_path"].values
    for id, image in images.items():
        if image.name in image_names:
            images_per_name[image.name] = image._replace(xys=[], point3D_ids=[])
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
