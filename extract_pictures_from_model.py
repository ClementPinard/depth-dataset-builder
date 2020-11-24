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
group = parser.add_mutually_exclusive_group()
group.add_argument('--metadata_path', metavar="CSV", type=Path, default=None,
                   help='Path to metadata CSV file of the desired video. '
                   'Usually in /pictures/Videos/<size>/<video_name>/metadata.csv')
group.add_argument('--picture_list_path', type=Path, default=None,
                   help='Path to list of picture to extract from model. '
                        'Picture paths must be relatvie to colmap root')


def extract_pictures(input, output, picture_list, output_format='.bin'):
    cameras = rm.read_cameras_binary(input / "cameras.bin")
    images = rm.read_images_binary(input / "images.bin")
    images_per_name = {}
    camera_ids = []

    def add_image(image):
        images_per_name[image.name] = image._replace(xys=[], point3D_ids=[])
        cam_id = image.camera_id
        if cam_id not in camera_ids:
            camera_ids.append(cam_id)

    for id, image in images.items():
        if image.name in picture_list:
            add_image(image)

    if len(images_per_name) == 1:
        # Add also first picture so that we have multiple pictures.
        # Otherwise, GourndTruth Creator will error
        for id, image in images.items():
            if image.name not in picture_list:
                add_image(image)
                break

    output_cameras = {cid: cameras[cid] for cid in camera_ids if cid in cameras.keys()}

    rm.write_model(output_cameras, images_per_name, {}, output, output_format)

    return len(images_per_name) > 1


def main():
    args = parser.parse_args()
    if args.metadata_path is not None:
        picture_list = pd.read_csv(args.metadata_path)["image_path"].values
    elif args.picture_list_path is not None:
        with open(args.picture_list_path, 'r') as f:
            picture_list = [line[:-1] for line in f.readlines()]
    extract_pictures(args.input_model, args.output_model, picture_list, args.output_format)
    return


if __name__ == '__main__':
    main()
