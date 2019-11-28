from pyproj import Proj
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
from tqdm import tqdm
from edit_exif import get_gps_location
import numpy as np


parser = ArgumentParser(description='extract XYZ data from exif and substract cloud centroid to register them',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--root', metavar='DIR',
                    help='path to video folder root')
parser.add_argument('--centroid_path', default=None)
parser.add_argument('--system', default='epsg:2154')
parser.add_argument('--output', metavar='PATH', default='images.txt')


def main():
    args = parser.parse_args()
    root = Path(args.root)
    if args.centroid_path is not None:
        centroid = np.loadtxt(args.centroid_path)
    else:
        centroid = np.zeros(3)

    proj = Proj(args.system)
    result = []
    folders = [root] + list(root.walkdirs())
    file_exts = ['jpg', 'JPG']

    for folder in tqdm(folders):

        current_folder = root.relpathto(folder)
        images = sum((folder.files('*{}'.format(ext)) for ext in file_exts), [])
        for img in images:
            loc = get_gps_location(img)
            if loc is None:
                continue
            lat, lon, alt = loc
            x, y = proj(lon, lat)
            pos = np.array([x, y, alt]) - centroid
            result.append('{} {} {} {}\n'.format(current_folder/img.basename(), *pos))

    with open(args.output, 'w') as f:
        f.writelines(result)


if __name__ == '__main__':
    main()
