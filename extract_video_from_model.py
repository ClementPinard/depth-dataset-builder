from colmap import read_model as rm
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path

parser = ArgumentParser(description='create a new colmap model with only the frames of selected video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_list', metavar='PATH',
                    help='path to list with relative path to images', type=Path)
parser.add_argument('--input_model', metavar='DIR', type=Path)
parser.add_argument('--output_model', metavar='DIR', default=None, type=Path)
parser.add_argument('--format', choices=['.txt', '.bin'], default='.txt')


def main():
    args = parser.parse_args()
    with open(args.video_list, 'r') as f:
        image_list = f.read().splitlines()
    cameras = rm.read_cameras_binary(args.input_model / "cameras.bin")
    images = rm.read_images_binary(args.input_model / "images.bin")
    images_per_name = {}
    for id, image in images.items():
        if image.name in image_list:
            images_per_name[image.name] = image
    camera_id = images_per_name[image_list[0]].camera_id
    cameras = {camera_id: cameras[camera_id]}

    rm.write_model(cameras, images_per_name, {}, args.output_model, args.format)
    return


if __name__ == '__main__':
    main()
