from colmap_util import read_model as rm, database as db
import anafi_metadata as am
from wrappers import FFMpeg, PDraw
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import pandas as pd
import numpy as np
from pyproj import Proj
from tqdm import tqdm

parser = ArgumentParser(description='Take all the drone videos of a folder and put the frame '
                                    'location in a COLMAP file for vizualisation',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_folder', metavar='DIR',
                    help='path to videos', type=Path)
parser.add_argument('--system', default='epsg:2154')
parser.add_argument('--centroid_path', default=None)
parser.add_argument('--output_folder', metavar='DIR', type=Path)
parser.add_argument('--workspace', metavar='DIR', type=Path)
parser.add_argument('--image_path', metavar='DIR', type=Path)
parser.add_argument('--output_format', metavar='EXT', default="bin")
parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"])
parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"])
parser.add_argument('--nw', default='',
                    help="native-wrapper.sh file location")
parser.add_argument('--fps', default=1, type=int,
                    help="framerate at which videos will be scanned WITH reconstruction")
parser.add_argument('--num_frames', default=200, type=int)
parser.add_argument('--orientation_weight', default=1, type=float)
parser.add_argument('--resolution_weight', default=1, type=float)
parser.add_argument('--num_neighbours', default=10, type=int)
parser.add_argument('--save_space', action="store_true")


def print_cams(cameras):
    print("id \t model \t \t width \t height \t params")
    for id, c in cameras.items():
        param_string = " ".join(["{:.3f}".format(p) for p in c.params])
        print("{} \t {} \t {} \t {} \t {}".format(id, c.model, c.width, c.height, param_string))


def world_coord_from_frame(frame_qvec, frame_tvec):
    '''
    frame_qvec is written in the NED system (north east down)
    frame_tvec is already is the world system (east norht up)
    '''
    world2NED = np.float32([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, -1]])
    NED2cam = np.float32([[0, 1, 0],
                          [0, 0, 1],
                          [1, 0, 0]])
    world2cam = NED2cam @ rm.qvec2rotmat(frame_qvec).T @ world2NED
    cam_tvec = - world2cam  @ frame_tvec
    cam_qvec = rm.rotmat2qvec(world2cam)
    return cam_qvec, cam_tvec


def optimal_sample(metadata, num_frames, orientation_weight, resolution_weight):
    metadata["sampled"] = False
    XYZ = metadata[["x", "y", "location_altitude"]].values
    axis_angle = metadata[["frame_quat_x", "frame_quat_y", "frame_quat_z"]].values
    if True in metadata["indoor"].unique():
        diameter = (XYZ.max(axis=0) - XYZ.min(axis=0))
        videos = metadata.loc[metadata["indoor"]]["video"].unique()
        new_centroids = 2 * diameter * np.linspace(0, 10, len(videos)).reshape(-1, 1)
        for centroid, v in zip(new_centroids, videos):
            video_index = (metadata["video"] == v).values
            XYZ[video_index] += centroid

    frame_size = metadata["video_quality"].values
    weighted_point_cloud = np.concatenate([XYZ, orientation_weight * axis_angle], axis=1)

    if resolution_weight == 0:
        weights = None
    else:
        weights = frame_size ** resolution_weight
    km = KMeans(n_clusters=num_frames).fit(weighted_point_cloud, sample_weight=weights)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, weighted_point_cloud)
    metadata.at[closest, "sampled"] = True
    return metadata


def register_new_cameras(cameras_dataframe, database, camera_dict, model_name="PINHOLE"):
    camera_ids = []
    for _, (w, h, f, hfov, vfov) in cameras_dataframe.iterrows():
        fx = w / (2 * np.tan(hfov * np.pi/360))
        fy = h / (2 * np.tan(vfov * np.pi/360))
        params = np.array([fx, fy, w/2, h/2])
        model_id = rm.CAMERA_MODEL_NAMES[model_name].model_id
        db_id = database.add_camera(model_id, w, h, params, prior_focal_length=True)
        camera_ids.append(db_id)
        camera_dict[db_id] = rm.Camera(id=db_id,
                                       model=model_name,
                                       width=int(w),
                                       height=int(h),
                                       params=params)
    ids_series = pd.Series(camera_ids)
    return cameras_dataframe.set_index(ids_series)


def process_video_folder(videos_list, existing_pictures, output_video_folder, image_path, system, centroid,
                         workspace, fps=1, total_frames=500, orientation_weight=1, resolution_weight=1,
                         output_colmap_format="bin", save_space=False, **env):
    proj = Proj(system)
    indoor_videos = []
    final_metadata = []
    video_output_folders = {}
    images = {}
    colmap_cameras = {}
    database_filepath = workspace/"thorough_scan.db"
    path_lists_output = {}
    database = db.COLMAPDatabase.connect(database_filepath)
    database.create_tables()
    to_extract = total_frames - len(existing_pictures)
    if to_extract <= 0:
        return None, None

    print("extracting metadata for {} videos...".format(len(videos_list)))
    for v in tqdm(videos_list):
        width, height, framerate = env["ffmpeg"].get_size_and_framerate(v)
        video_output_folder = output_video_folder / "{}x{}".format(width, height) / v.namebase
        video_output_folder.makedirs_p()
        video_output_folders[v] = video_output_folder

        metadata = am.extract_metadata(v.parent, v, env["pdraw"], proj,
                                       width, height, framerate, centroid)
        final_metadata.append(metadata)
        if metadata["indoor"].iloc[0]:
            indoor_videos.append(v)
    final_metadata = pd.concat(final_metadata, ignore_index=True)
    print("{} outdoor videos".format(len(videos_list) - len(indoor_videos)))
    print("{} indoor videos".format(len(indoor_videos)))

    print("{} frames in total".format(len(final_metadata)))

    cam_fields = ["width", "height", "framerate", "picture_hfov", "picture_vfov"]
    cameras_dataframe = final_metadata[cam_fields].drop_duplicates()
    cameras_dataframe = register_new_cameras(cameras_dataframe, database, colmap_cameras, "PINHOLE")
    print(cameras_dataframe)
    final_metadata["camera_id"] = 0
    for cam_id, row in cameras_dataframe.iterrows():
        final_metadata.loc[(final_metadata[cam_fields] == row).all(axis=1), "camera_id"] = cam_id
    if any(final_metadata["camera_id"] == 0):
        print("Error")
        print((final_metadata["camera_id"] == 0))

    if to_extract <= len(final_metadata):
        print("subsampling based on K-Means, to get {}"
              " frames from videos, for a total of {} frames".format(to_extract, total_frames))
        final_metadata = optimal_sample(final_metadata, total_frames - len(existing_pictures),
                                        orientation_weight,
                                        resolution_weight)
        print("Done.")

    print("Constructing COLMAP model with {:,} frames".format(len(final_metadata[final_metadata["sampled"]])))

    final_metadata["image_path"] = ""
    for image_id, row in tqdm(final_metadata.iterrows(), total=len(final_metadata)):
        video = row["video"]
        frame = row["frame"]
        camera_id = row["camera_id"]
        current_image_path = video_output_folders[video].relpath(image_path) / video.namebase + "_{:05d}.jpg".format(frame)

        final_metadata.at[image_id, "image_path"] = current_image_path

        if row["sampled"]:
            frame_qvec = row[["frame_quat_w",
                              "frame_quat_x",
                              "frame_quat_y",
                              "frame_quat_z"]].values
            x, y, alt = row["x"], row["y"], row["location_altitude"]
            frame_tvec = np.array([x, y, alt])
            if row["location_valid"]:
                frame_gps = row[["location_longitude", "location_latitude", "location_altitude"]]
            else:
                frame_gps = np.full(3, np.NaN)

            world_qvec, world_tvec = world_coord_from_frame(frame_qvec, frame_tvec)
            db_image_id = database.add_image(current_image_path, int(camera_id), prior_t=frame_gps)
            images[db_image_id] = rm.Image(
                id=db_image_id, qvec=world_qvec, tvec=world_tvec,
                camera_id=camera_id, name=current_image_path,
                xys=[], point3D_ids=[])

    database.commit()
    database.close()
    rm.write_model(colmap_cameras, images, {}, output_video_folder, "." + output_colmap_format)
    print("COLMAP model created")

    thorough_scan_images = final_metadata[final_metadata["sampled"]][["image_path", "x", "y", "location_altitude"]]
    path_lists_output["thorough"] = []
    path_lists_output["georef"] = []

    for _, (path, x, y, alt) in thorough_scan_images.iterrows():
        path_lists_output["thorough"].append(path)
        path_lists_output["georef"].append("{} {} {} {}\n".format(path, x, y, alt))

    print("Extracting frames from videos")

    for v in tqdm(videos_list):
        video_metadata = final_metadata[final_metadata["video"] == v]
        image_paths = video_metadata.set_index(pd.to_datetime(video_metadata["time"], unit="us"))["image_path"]
        video_folder = video_output_folders[v]
        video_metadata.to_csv(video_folder/"metadata.csv")
        per_video_scan_image_paths = image_paths.resample("{:.3f}S".format(1/fps)).first()
        path_lists_output[v] = list(per_video_scan_image_paths)
        if save_space:
            frame_ids = list(video_metadata[video_metadata["sampled"]]["frame"].values)
            env["ffmpeg"].extract_specific_frames(v, video_folder, frame_ids)
        else:
            env["ffmpeg"].extract_images(v, video_folder)

    return path_lists_output, video_output_folders


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    env["videos_list"] = sum((list(args.video_folder.walkfiles('*{}'.format(ext))) for ext in args.vid_ext), [])
    args.workspace.makedirs_p()
    output_video_folder = args.output_folder / "Videos"
    output_video_folder.makedirs_p()
    env["image_path"] = args.output_folder
    env["output_video_folder"] = output_video_folder
    env["existing_pictures"] = sum((list(args.output_folder.walkfiles('*{}'.format(ext))) for ext in args.pic_ext), [])
    env["pdraw"] = PDraw(args.nw, quiet=True)
    env["ffmpeg"] = FFMpeg(quiet=True)

    if args.centroid_path is not None:
        centroid = np.loadtxt(args.centroid_path)
    else:
        centroid = np.zeros(3)
    env["centroid"] = centroid
    lists, extracted_video_folders = process_video_folder(**env)

    if lists is not None:
        with open(args.output_folder/"video_frames_for_thorough_scan.txt", "w") as f:
            f.write("\n".join(lists["thorough"]))
        with open(args.output_folder/"georef.txt", "w") as f:
            f.write("\n".join(lists["georef"]))
        for v in env["videos_list"]:
            with open(extracted_video_folders[v] / "to_scan.txt", "w") as f:
                f.write("\n".join(lists[v]))
