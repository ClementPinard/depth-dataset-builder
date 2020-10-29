import pandas as pd
import numpy as np
from path import Path
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from colmap.read_model import Image, Camera, Point3D, write_model, qvec2rotmat, rotmat2qvec
from tqdm import tqdm
from pyntcloud import PyntCloud
from itertools import islice

parser = ArgumentParser(description='Convert EuroC dataset to COLMAP',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--root', metavar='DIR', type=Path)
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--img_root', metavar='DIR', default=None, type=Path)
parser.add_argument('--load_pointcloud', action='store_true')
parser.add_argument('--format', choices=['.txt', '.bin'], default='.txt')


def get_cam(yaml_path, cam_id):
    with open(yaml_path) as f:
        cam_dict = yaml.load(f, Loader=yaml.SafeLoader)

    calib = cam_dict["T_BS"]
    calib_matrix = np.array(calib["data"]).reshape((calib["rows"], calib["cols"]))
    assert cam_dict["distortion_model"] == "radial-tangential"
    w, h = cam_dict["resolution"]
    cam = Camera(id=cam_id,
                 model="OPENCV",
                 width=w,
                 height=h,
                 params=np.array(cam_dict["intrinsics"] + cam_dict["distortion_coefficients"]))

    return cam, calib_matrix


def get_vicon_calib(yaml_path):
    with open(yaml_path) as f:
        vicon_dict = yaml.load(f, Loader=yaml.SafeLoader)

    calib = vicon_dict["T_BS"]
    return np.array(calib["data"]).reshape((calib["rows"], calib["cols"]))


def create_image(img_id, cam_id, file_path, drone_pose, image_calib, vicon_calib):
    t_prefix = " p_RS_R_{} [m]"
    q_prefix = " q_RS_{} []"
    drone_tvec = drone_pose[[t_prefix.format(dim) for dim in 'xyz']].to_numpy().reshape(3, 1)
    drone_qvec = drone_pose[[q_prefix.format(dim) for dim in 'wxyz']].to_numpy()
    drone_R = qvec2rotmat(drone_qvec)
    drone_matrix = np.concatenate((np.hstack((drone_R, drone_tvec)), np.array([0, 0, 0, 1]).reshape(1, 4)))
    image_matrix = drone_matrix @ np.linalg.inv(vicon_calib) @ image_calib
    colmap_matrix = np.linalg.inv(image_matrix)
    colmap_qvec = rotmat2qvec(colmap_matrix[:3, :3])
    colmap_tvec = colmap_matrix[:3, -1]

    return Image(id=img_id, qvec=colmap_qvec, tvec=colmap_tvec,
                 camera_id=cam_id, name=file_path,
                 xys=[], point3D_ids=[]), image_matrix[:3, -1]


def main():
    args = parser.parse_args()
    cam_dirs = [args.root/"cam0", args.root/"cam1"]
    vicon_dir = args.root/"state_groundtruth_estimate0"
    cloud_file = args.root/"pointcloud0"/"data.ply"
    if args.img_root is None:
        args.img_root = args.root

    vicon_poses = pd.read_csv(vicon_dir/"data.csv")
    vicon_poses = vicon_poses.set_index("#timestamp")
    vicon_calib = get_vicon_calib(vicon_dir/"sensor.yaml")
    cameras = {}
    images = {}
    image_list = []
    image_georef = []
    for cam_id, cam in enumerate(cam_dirs):
        print("Converting camera {} ...".format(cam))
        if len(images.keys()) == 0:
            last_image_id = 0
        else:
            last_image_id = max(images.keys())
        cameras[cam_id], cam_calib = get_cam(cam/"sensor.yaml", cam_id)

        image_names = pd.read_csv(cam/"data.csv")
        image_root = cam/"data"
        step = 1
        for img_id, (_, (ts, filename)) in tqdm(enumerate(islice(image_names.iterrows(), 0, None, step)), total=len(image_names.index)//step):
            final_path = (image_root/filename).relpath(args.img_root)
            image_list.append(final_path)
            row_index = vicon_poses.index.get_loc(ts, method='nearest')
            current_drone_pose = vicon_poses.iloc[row_index]
            images[1 + img_id + last_image_id], georef = create_image(1 + img_id + last_image_id, cam_id,
                                                                      final_path, current_drone_pose,
                                                                      cam_calib, vicon_calib)
            image_georef.append(georef)

    points = {}
    if args.load_pointcloud:
        subsample = 1
        print("Loading point cloud {}...".format(cloud_file))
        cloud = PyntCloud.from_file(cloud_file)
        print("Converting ...")
        npy_points = cloud.points[['x', 'y', 'z', 'intensity']].values[::subsample]
        for id_point, row in tqdm(enumerate(npy_points), total=len(npy_points)):
            xyz = row[:3]
            gray_level = int(row[-1]*255)
            rgb = np.array([gray_level] * 3)
            points[id_point] = Point3D(id=id_point, xyz=xyz, rgb=rgb,
                                       error=0, image_ids=np.array([]),
                                       point2D_idxs=np.array([]))
    with open(args.root/"images.txt", "w") as f1, open(args.root/"georef.txt", "w") as f2:
        for path, pos in zip(image_list, image_georef):
            f1.write(path + "\n")
            f2.write("{} {} {} {}\n".format(path, *pos))
    write_model(cameras, images, points, args.output_dir, args.format)


if __name__ == '__main__':
    main()
