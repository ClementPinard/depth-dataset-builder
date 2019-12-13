from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path

parser = ArgumentParser(description='Create vizualisation for specified video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--gt', metavar='PATH',
                    help='path to folder with raw groundtruth', type=Path)
parser.add_argument('--images', metavar='PATH',
                    help='path to folder with images', type=Path)
parser.add_argument('--range', metavar='R', default=10, type=int,
                    help='')
parser.add_argument('--output', metavar='DIR', default=None, type=Path)


def main():
    args = parser.parse_args()
    depth_paths = sorted(args.gt.files())
    image_paths = sorted(args.images.files())

    if len(depth_paths) != len(image_paths):
        print("{} depth groundtruth maps for {} images".format(len(depth_paths), len(image_paths)))


if __name__ == '__main__':
    main()
