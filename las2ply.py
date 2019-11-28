from pyntcloud import PyntCloud
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np


parser = ArgumentParser(description='Convert las cloud to ply along with centroid',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('las',
                    help='path to video folder root')
parser.add_argument('--output_ply', metavar='PATH', default=None)
parser.add_argument('--output_txt', metavar='PATH', default=None)


def main():
    args = parser.parse_args()
    las_path = Path(args.las)

    if args.output_ply is None:
        ply_path = las_path.stripext() + '_converted.ply'
    else:
        ply_path = args.output_ply

    if args.output_txt is None:
        txt_path = las_path.stripext() + '_centroid.txt'
    else:
        txt_path = args.output_txt

    cloud = PyntCloud.from_file(las_path)
    print("Las file with {:,} points "
          "(centroid : [{:.2f}, {:.2f}, {:.2f}] in km) "
          "successfully loaded".format(len((cloud.points)), *(cloud.centroid/1000)))

    np.savetxt(txt_path, cloud.centroid)

    xyz = cloud.points[['x', 'y', 'z']]
    cloud.points = xyz
    cloud.points -= cloud.centroid

    cloud.to_file(ply_path)

    print("saved shifted cloud to {}, centroid to {}".format(ply_path, txt_path))


if __name__ == '__main__':
    main()
