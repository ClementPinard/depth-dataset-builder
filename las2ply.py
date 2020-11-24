import laspy
from pyntcloud import PyntCloud
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np


parser = ArgumentParser(description='Convert las cloud to ply along with centroid',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('las', type=Path,
                    help='Path to las file. Note that this script is compatible with PLY')
parser.add_argument('--output_folder', metavar='PATH',
                    default=None, type=Path,
                    help="where to save ply file and txt centroid")
parser.add_argument('--verbose', '-v', action='store_true')


def load_and_convert(input_file, output_folder, verbose=False):
    output_folder.makedirs_p()
    ply_path = output_folder / input_file.stem + '.ply'
    txt_path = output_folder / input_file.stem + '_centroid.txt'
    file_type = input_file.ext[1:].upper()
    if file_type == "LAS":
        offset = np.array(laspy.file.File(input_file, mode="r").header.offset)
    else:
        offset = np.zeros(3)
    cloud = PyntCloud.from_file(input_file)
    if verbose:
        print(cloud.points)

    points = cloud.points
    xyz = points[['x', 'y', 'z']]
    xyz += offset
    points[['x', 'y', 'z']] = xyz
    cloud.points = points

    if verbose:
        print("{} file with {:,} points "
              "(centroid : [{:.2f}, {:.2f}, {:.2f}] in km) "
              "successfully loaded".format(file_type,
                                           len((cloud.points)),
                                           *(cloud.centroid/1000)))

    output_centroid = cloud.centroid
    np.savetxt(txt_path, output_centroid)

    points = cloud.points
    xyz = points[['x', 'y', 'z']]
    xyz -= cloud.centroid
    points[['x', 'y', 'z']] = xyz
    if (all([c in points.keys() for c in ["red", "green", "blue"]])):
        points[['red', 'green', 'blue']] = (points[['red', 'green', 'blue']] / 255).astype(np.uint8)
        invalid_color = (points["red"] > 250) & (points["green"] > 250) & (points["blue"] > 250)
        cloud.points = points[["x", "y", "z", "red", "green", "blue"]][~invalid_color]

    cloud.to_file(ply_path)
    if verbose:
        print("saved shifted cloud to {}, centroid to {}".format(ply_path, txt_path))
    return ply_path, output_centroid


if __name__ == '__main__':
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = args.las_path.parent
    load_and_convert(args.las, args.output_folder, args.verbose)
