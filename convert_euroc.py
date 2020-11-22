import pandas as pd
import numpy as np
from path import Path
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from colmap_util.read_model import Image, Camera, Point3D, write_model, rotmat2qvec
import meshlab_xml_writer as mxw
from tqdm import tqdm
from pyntcloud import PyntCloud
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from wrappers import FFMpeg

parser = ArgumentParser(description='Convert EuroC dataset to COLMAP',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--root', metavar='DIR', type=Path, help='path to root folder eof EuRoC, where V[N]_[M]_[difficulty] folders should be')
parser.add_argument('--output_dir', metavar='DIR', default=None, type=Path)
parser.add_argument('--pointcloud_to_colmap', action='store_true')
parser.add_argument('--colmap_format', choices=['.txt', '.bin'], default='.txt')
parser.add_argument("--ffmpeg", default="ffmpeg", type=Path)
parser.add_argument('--log', default=None, type=Path)
parser.add_argument('-v', '--verbose', action="count", default=0)


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


def create_image(img_id, cam_id, file_path, drone_tvec, drone_matrix, image_calib, vicon_calib):
    drone_full_matrix = np.concatenate((np.hstack((drone_matrix, drone_tvec[:, None])), np.array([0, 0, 0, 1]).reshape(1, 4)))
    image_matrix = drone_full_matrix @ np.linalg.inv(vicon_calib) @ image_calib
    colmap_matrix = np.linalg.inv(image_matrix)
    colmap_qvec = rotmat2qvec(colmap_matrix[:3, :3])
    colmap_tvec = colmap_matrix[:3, -1]

    return Image(id=img_id, qvec=colmap_qvec, tvec=colmap_tvec,
                 camera_id=cam_id, name=file_path,
                 xys=[], point3D_ids=[]), image_matrix[:3, -1]


def convert_cloud(input_dir, output_dir):
    cloud_path = input_dir / "data.ply"
    if not cloud_path.isfile():
        return None
    cloud = PyntCloud.from_file(cloud_path)
    cloud.points = cloud.points[['x', 'y', 'z', 'intensity']]
    yaml_path = input_dir / "sensor.yaml"
    with open(yaml_path) as f:
        cloud_dict = yaml.load(f, Loader=yaml.SafeLoader)
    calib = cloud_dict["T_WR"]
    transform = np.array(calib["data"]).reshape((calib["rows"], calib["cols"]))
    output_ply = output_dir / "data.ply"
    mxw.create_project(output_dir / 'data.mlp', [output_ply], labels=None, transforms=[transform])
    cloud.to_file(output_ply)
    return cloud


def main():
    args = parser.parse_args()
    scenes = ["V1", "V2"]
    ffmpeg = FFMpeg(args.ffmpeg, verbose=args.verbose, logfile=args.log)
    for s in scenes:
        pointcloud = None
        lidar_output = args.output_dir / s / "Lidar"
        video_output = args.output_dir / s / "Videos"
        lidar_output.makedirs_p()
        video_output.makedirs_p()
        (args.output_dir / s / "Pictures").makedirs_p()

        colmap_model = {"cams": {},
                        "imgs": {},
                        "points": {}}
        video_sequences = sorted(args.root.dirs("{}*".format(s)))
        cam_id = 0
        for v in video_sequences:
            mav = v / "mav0"
            cam_dirs = [mav/"cam0", mav/"cam1"]
            vicon_dir = mav/"state_groundtruth_estimate0"
            if pointcloud is None:
                cloud = convert_cloud(mav/"pointcloud0", lidar_output)

            vicon_poses = pd.read_csv(vicon_dir/"data.csv")
            vicon_poses = vicon_poses.set_index("#timestamp")
            min_ts, max_ts = min(vicon_poses.index), max(vicon_poses.index)
            t_prefix = " p_RS_R_{} [m]"
            q_prefix = " q_RS_{} []"
            drone_tvec = vicon_poses[[t_prefix.format(dim) for dim in 'xyz']].values
            drone_qvec = Rotation.from_quat(vicon_poses[[q_prefix.format(dim) for dim in 'xyzw']].values)
            drone_qvec_slerp = Slerp(vicon_poses.index, drone_qvec)
            drone_tvec_interp = interp1d(vicon_poses.index, drone_tvec.T)
            vicon_calib = get_vicon_calib(vicon_dir/"sensor.yaml")
            for cam in cam_dirs:
                output_video_file = video_output/"{}_{}.mp4".format(v.stem, cam.stem)
                image_georef = []
                image_rel_paths = []
                image_ids = []
                qvecs = []
                print("Converting camera {} from video {}...".format(cam.relpath(v), v))
                if len(colmap_model["imgs"].keys()) == 0:
                    last_image_id = 0
                else:
                    last_image_id = max(colmap_model["imgs"].keys()) + 1
                colmap_cam, cam_calib = get_cam(cam/"sensor.yaml", cam_id)
                colmap_model["cams"][cam_id] = colmap_cam
                metadata = pd.read_csv(cam/"data.csv").sort_values(by=['#timestamp [ns]'])
                metadata["camera_model"] = "OPENCV"
                metadata["width"] = colmap_cam.width
                metadata["height"] = colmap_cam.height
                metadata["camera_params"] = [tuple(colmap_cam.params)] * len(metadata)
                metadata["time"] = metadata['#timestamp [ns]']
                metadata = metadata[(metadata['time'] > min_ts) & (metadata['time'] < max_ts)]
                tvec_interpolated = drone_tvec_interp(metadata['time']).T
                qvec_interpolated = drone_qvec_slerp(metadata['time'])
                # Convert time from nanoseconds to microseconds for compatibility
                metadata['time'] = metadata['time'] * 1e-3
                for img_id, (filename, current_tvec, current_qvec) in tqdm(enumerate(zip(metadata["filename"].values,
                                                                                         tvec_interpolated,
                                                                                         qvec_interpolated)),
                                                                           total=len(metadata)):
                    final_path = args.root.relpathto(cam / "data") / filename
                    image_rel_paths.append(final_path)
                    colmap_model["imgs"][img_id + last_image_id], georef = create_image(img_id + last_image_id, cam_id,
                                                                                        final_path, current_tvec,
                                                                                        current_qvec.as_matrix(),
                                                                                        cam_calib, vicon_calib)
                    image_georef.append(georef)
                    image_ids.append(img_id + last_image_id)
                    qvecs.append(current_qvec.as_quat())

                metadata['x'], metadata['y'], metadata['z'] = np.array(image_georef).transpose()
                qvecs_array = np.array(qvecs).transpose()
                for coord, title in zip(qvecs_array, 'xyzw'):
                    metadata['frame_quat_{}'.format(title)] = coord
                metadata['image_path'] = image_rel_paths
                metadata['location_valid'] = True
                metadata['indoor'] = True
                metadata['video'] = cam
                framerate = len(metadata) / np.ptp(metadata['time'].values * 1e-6)
                metadata['framerate'] = framerate
                # Copy images for ffmpeg
                for i, f in enumerate(metadata["filename"]):
                    (cam / "data" / f).copy(video_output / "tmp_{:05d}.png".format(i))
                glob_pattern = str(video_output / "tmp_%05d.png")
                ffmpeg.create_video(output_video_file, glob_pattern, fps=framerate, glob=False)
                frames_to_delete = video_output.files("tmp*")
                for f in frames_to_delete:
                    f.remove()
                # Save metadata in csv file
                metadata_file_path = output_video_file.parent / "{}_metadata.csv".format(output_video_file.stem)
                metadata.to_csv(metadata_file_path)
                cam_id += 1

        points = {}
        if args.pointcloud_to_colmap and cloud is not None:
            subsample = 1
            print("Converting ...")
            npy_points = cloud.points[['x', 'y', 'z', 'intensity']].values[::subsample]
            for id_point, row in tqdm(enumerate(npy_points), total=len(npy_points)):
                xyz = row[:3]
                gray_level = int(row[-1]*255)
                rgb = np.array([gray_level] * 3)
                points[id_point] = Point3D(id=id_point, xyz=xyz, rgb=rgb,
                                           error=0, image_ids=np.array([]),
                                           point2D_idxs=np.array([]))
        with open(args.output_dir/"images.txt", "w") as f1, open(args.root/"georef.txt", "w") as f2:
            for path, pos in zip(image_rel_paths, image_georef):
                f1.write(path + "\n")
                f2.write("{} {} {} {}\n".format(path, *pos))
        colmap_output = args.output_dir / s / "colmap_from_GT"
        colmap_output.makedirs_p()
        write_model(colmap_model["cams"],
                    colmap_model["imgs"],
                    colmap_model["points"],
                    colmap_output,
                    args.colmap_format)


if __name__ == '__main__':
    main()
