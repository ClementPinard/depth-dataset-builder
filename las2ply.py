from pyntcloud import PyntCloud
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np


parser = ArgumentParser(description='Convert las cloud to ply along with centroid',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('las', type=Path,
                    help='path to video folder root')
parser.add_argument('--output_folder', metavar='PATH',
                    default=None, type=Path,
                    help="where to save ply file and txt centroid")
parser.add_argument('--verbose', '-v', action='store_true')


def load_and_convert(input_file, output_folder, verbose=False):
    output_folder.makedirs_p()
    ply_path = output_folder / input_file.namebase + '.ply'
    txt_path = output_folder / input_file.namebase + '_centroid.txt'
    cloud = PyntCloud.from_file(input_file)
    file_type = input_file.ext[1:].upper()
    if verbose:
        print("{} file with {:,} points "
              "(centroid : [{:.2f}, {:.2f}, {:.2f}] in km) "
              "successfully loaded".format(file_type,
                                           len((cloud.points)),
                                           *(cloud.centroid/1000)))

    output_centroid = cloud.centroid
    np.savetxt(txt_path, output_centroid)

    xyz = cloud.points[['x', 'y', 'z']]
    cloud.points = xyz
    cloud.points -= cloud.centroid

    cloud.to_file(ply_path)
    if verbose:
        print("saved shifted cloud to {}, centroid to {}".format(ply_path, txt_path))
    return ply_path, output_centroid


if __name__ == '__main__':
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = args.las_path.parent
    load_and_convert(args.las_path, args.output_folder, args.verbose)
