from colmap_util import read_model as rm
from colmap_util.database import COLMAPDatabase
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd
import numpy as np
from pyproj import Proj

parser = ArgumentParser(description='Add GPS localized video to colmap model (Note : Localization is not precise enough)',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_list', metavar='PATH',
                    help='path to list with relative path to images', type=Path)
parser.add_argument('--metadata', metavar='PATH',
                    help='path to metadata csv file', type=Path)
parser.add_argument('--database', metavar='DB', required=True,
                    help='path to colmap database file, to get the image ids right')
parser.add_argument('--input_model', metavar='Path', type=Path)
parser.add_argument('--system', default='epsg:2154')
parser.add_argument('--centroid_path', default=None)
parser.add_argument('--output_model', metavar='DIR', default=None, type=Path)


def print_cams(cameras):
    print("id \t model \t \t width \t height \t params")
    for id, c in cameras.items():
        param_string = " ".join(["{:.3f}".format(p) for p in c.params])
        print("{} \t {} \t {} \t {} \t {}".format(id, c.model, c.width, c.height, param_string))


def print_imgs(images, max_img=2):
    max_img = min(max_img, len(images))
    keys = sorted(images.keys())[:max_img]
    for k in keys:
        print(images[k])


def world_coord_from_frame(frame_qvec, frame_tvec):
    world2NED = np.float32([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
    NED2cam = np.float32([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]])
    frame_rot = NED2cam @ rm.qvec2rotmat(frame_qvec).T @ world2NED
    world_tvec = - frame_rot @ frame_tvec
    world_qvec = rm.rotmat2qvec(frame_rot)
    return world_qvec, world_tvec


def get_id_from_db(db):
    rows = db.execute("SELECT * FROM images")
    id_name = {}
    for id, name, *_ in rows:
        id_name[name] = id
    return id_name


def main():
    args = parser.parse_args()
    proj = Proj(args.system)
    if args.centroid_path is not None:
        centroid = np.loadtxt(args.centroid_path)
    else:
        centroid = np.zeros(3)
    db = COLMAPDatabase.connect(args.database)
    with open(args.video_list, 'r') as f:
        image_list = f.read().splitlines()
    cameras = rm.read_cameras_binary(args.input_model / "cameras.bin")
    print("Available cameras :")
    print_cams(cameras)
    camera_id = int(input("which camera for the video ?\n"))
    images = rm.read_images_binary(args.input_model / "images.bin")
    # images = {}
    image_ids = get_id_from_db(db)
    for name in image_list:
        if name not in image_ids.keys():
            raise Exception("Image {} not in database".format(name))

    metadata = pd.read_csv(args.metadata, sep=" ")
    for (i, row), image_path in zip(metadata.iterrows(), image_list):
        image_id = image_ids[image_path]
        frame_qvec = np.array([row["frame_quat_w"],
                               row["frame_quat_x"],
                               row["frame_quat_y"],
                               row["frame_quat_z"]])
        lat, lon, alt = row["location_latitude"], row["location_longitude"], row["location_altitude"]
        x, y = proj(lon, lat)
        frame_tvec = np.array([x, y, alt]) - centroid
        world_qvec, world_tvec = world_coord_from_frame(frame_qvec, frame_tvec)
        images[image_id] = rm.Image(
            id=image_id, qvec=world_qvec, tvec=world_tvec,
            camera_id=camera_id, name=image_path,
            xys=[], point3D_ids=[])

    rm.write_images_binary(images, args.output_model, "images.bin")
    rm.write_points3d_binary({}, args.output_model / "points3D.bin")
    return


if __name__ == '__main__':
    main()
