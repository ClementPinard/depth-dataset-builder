from colmap_util import read_model as rm, database as db
import anafi_metadata as am
from wrappers import FFMpeg, PDraw
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from edit_exif import set_gps_location
from path import Path
import pandas as pd
import numpy as np
from pyproj import Proj
from tqdm import tqdm
import tempfile

parser = ArgumentParser(description='Take all the drone videos of a folder and put the frame '
                                    'location in a COLMAP file for vizualisation',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_folder', metavar='DIR',
                    help='path to videos', type=Path)
parser.add_argument('--system', default='epsg:2154')
parser.add_argument('--centroid_path', default=None)
parser.add_argument('--colmap_img_root', metavar='DIR', type=Path)
parser.add_argument('--output_format', metavar='EXT', default="bin")
parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"])
parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"])
parser.add_argument('--nw', default='',
                    help="native-wrapper.sh file location")
parser.add_argument('--fps', default=1, type=int,
                    help="framerate at which videos will be scanned WITH reconstruction")
parser.add_argument('--total_frames', default=200, type=int)
parser.add_argument('--orientation_weight', default=1, type=float)
parser.add_argument('--resolution_weight', default=1, type=float)
parser.add_argument('--save_space', action="store_true")
parser.add_argument('--thorough_db', type=Path)
parser.add_argument('-v', '--verbose', action="count", default=0)


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


def set_gps(frames_list, metadata, image_path):
    for frame in frames_list:
        relative = str(frame.relpath(image_path))
        row = metadata[metadata["image_path"] == relative]
        if len(row) > 0:
            row = row.iloc[0]
            if row["location_valid"]:
                set_gps_location(frame,
                                 lat=row["location_latitude"],
                                 lng=row["location_longitude"],
                                 altitude=row["location_altitude"])


def get_georef(metadata):
    relevant_data = metadata[["location_valid", "image_path", "x", "y", "z"]]
    path_list = []
    georef_list = []
    for _, (gps, path, x, y, alt) in relevant_data.iterrows():
        path_list.append(path)
        if gps == 1:
            georef_list.append("{} {} {} {}\n".format(path, x, y, alt))
    return georef_list, path_list


def optimal_sample(metadata, num_frames, orientation_weight, resolution_weight):
    valid_metadata = metadata[~metadata["sampled"]].dropna()
    if len(valid_metadata) == 0:
        return metadata
    XYZ = valid_metadata[["x", "y", "z"]].values
    axis_angle = valid_metadata[["frame_quat_x", "frame_quat_y", "frame_quat_z"]].values
    if True in valid_metadata["indoor"].unique():
        # We have indoor videos, without absolute positions. We assume each video is very far
        # from the other ones. As such we will have an optimal subsampling of each video
        # It won't leverage video proximity from each other but it's better than nothing
        diameter = (XYZ.max(axis=0) - XYZ.min(axis=0))
        indoor_videos = valid_metadata.loc[valid_metadata["indoor"]]["video"].unique()
        new_centroids = 2 * diameter * np.linspace(0, 10, len(indoor_videos)).reshape(-1, 1)
        for centroid, v in zip(new_centroids, indoor_videos):
            video_index = (valid_metadata["video"] == v).values
            XYZ[video_index] += centroid

    frame_size = valid_metadata["video_quality"].values
    weighted_point_cloud = np.concatenate([XYZ, orientation_weight * axis_angle], axis=1)

    if resolution_weight == 0:
        weights = None
    else:
        weights = frame_size ** resolution_weight
    km = KMeans(n_clusters=num_frames).fit(weighted_point_cloud, sample_weight=weights)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, weighted_point_cloud)
    metadata.at[valid_metadata.index[closest], "sampled"] = True
    return metadata


def register_new_cameras(cameras_dataframe, database, camera_dict):
    camera_ids = []
    for _, row in cameras_dataframe.iterrows():
        w, h, hfov, vfov, camera_model = row.reindex(["width", "height", "picture_hfov", "picture_vfov", "camera_model"])
        prior_focal_length = False
        single_focal = ('SIMPLE' in camera_model) or ('RADIAL' in camera_model)
        if hfov != 0:
            fx = w / (2 * np.tan(hfov * np.pi/360))
            # If the model is not single focal, only knowing hfov is not enough, you also need to know vfov
            prior_focal_length = single_focal
        else:
            fx = w / 2  # As if hfov was 90 degrees
        if vfov != 0:
            fy = h / (2 * np.tan(vfov * np.pi/360))
            prior_focal_length = True
        else:
            fy = w / 2  # As if vfov was 90 degrees
        model_id = rm.CAMERA_MODEL_NAMES[camera_model].model_id
        num_params = rm.CAMERA_MODEL_NAMES[camera_model].num_params
        if ('SIMPLE' in camera_model) or ('RADIAL' in camera_model):
            params = np.array([fx, w/2, h/2] + [0] * (num_params - 3))
        else:
            params = np.array([fx, fy, w/2, h/2] + [0] * (num_params - 4))
        db_id = database.add_camera(model_id, int(w), int(h), params, prior_focal_length=prior_focal_length)
        camera_ids.append(db_id)
        camera_dict[db_id] = rm.Camera(id=db_id,
                                       model=camera_model,
                                       width=int(w),
                                       height=int(h),
                                       params=params)
    ids_series = pd.Series(camera_ids)
    return cameras_dataframe.set_index(ids_series)


def process_video_folder(videos_list, existing_pictures, output_video_folder, image_path, system, centroid,
                         thorough_db, fps=1, total_frames=500, orientation_weight=1, resolution_weight=1,
                         output_colmap_format="bin", save_space=False, include_lowfps_thorough=False,
                         max_sequence_length=1000, **env):
    proj = Proj(system)
    final_metadata = []
    video_output_folders = {}
    images = {}
    colmap_cameras = {}
    tempfile_database = Path(tempfile.NamedTemporaryFile().name)
    if thorough_db.isfile():
        thorough_db.copy(thorough_db.stripext() + "_backup.db")
    path_lists_output = {}
    database = db.COLMAPDatabase.connect(thorough_db)
    database.create_tables()

    print("extracting metadata for {} videos...".format(len(videos_list)))
    videos_summary = {"anafi": {"indoor": 0, "outdoor": 0}, "generic": 0}
    for v in tqdm(videos_list):
        width, height, framerate, num_frames = env["ffmpeg"].get_size_and_framerate(v)
        video_output_folder = output_video_folder / "{}x{}".format(width, height) / v.stem
        video_output_folder.makedirs_p()
        video_output_folders[v] = video_output_folder

        try:
            metadata = am.extract_metadata(v.parent, v, env["pdraw"], proj,
                                           width, height, framerate)
            metadata["model"] = "anafi"
            metadata["camera_model"] = "PINHOLE"
            if metadata["indoor"].iloc[0]:
                videos_summary["anafi"]["indoor"] += 1
            else:
                videos_summary["anafi"]["outdoor"] += 1
                raw_positions = metadata[["x", "y", "z"]]
                if centroid is None:
                    '''No centroid (possibly because there was no georeferenced lidar model in the first place)
                    set it as the first valid GPS position of the first outdoor video'''
                    centroid = raw_positions[metadata["location_valid"] == 1].iloc[0].values
                zero_centered_positions = raw_positions.values - centroid
                radius = np.max(np.abs(zero_centered_positions))
                if radius > 1000:
                    print("Warning, your positions coordinates are most likely too high, have you configured the right GPS system ?")
                    print("It should be the same as the one used for the Lidar point cloud")
                metadata["x"], metadata["y"], metadata["z"] = zero_centered_positions.transpose()
        except Exception:
            # No metadata found, construct a simpler dataframe without location
            metadata = pd.DataFrame({"video": [v] * num_frames})
            metadata["height"] = height
            metadata["width"] = width
            metadata["framerate"] = framerate
            metadata["video_quality"] = height * width / framerate
            metadata['frame'] = metadata.index + 1
            # timestemp is in microseconds
            metadata['time'] = 1e6 * metadata.index / framerate
            metadata['indoor'] = True
            metadata['location_valid'] = 0
            metadata["model"] = "generic"
            metadata["camera_model"] = "PINHOLE"
            metadata["picture_hfov"] = 0
            metadata["picture_vfov"] = 0
            metadata["frame_quat_w"] = np.NaN
            metadata["frame_quat_x"] = np.NaN
            metadata["frame_quat_y"] = np.NaN
            metadata["frame_quat_z"] = np.NaN
            metadata["x"] = np.NaN
            metadata["y"] = np.NaN
            metadata["z"] = np.NaN
            videos_summary["generic"] += 1
        if include_lowfps_thorough:
            by_time = metadata.set_index(pd.to_datetime(metadata["time"], unit="us"))
            by_time_lowfps = by_time.resample("{:.3f}S".format(1/fps)).first()
            metadata["sampled"] = by_time["time"].isin(by_time_lowfps["time"]).values
        else:
            metadata["sampled"] = False
        final_metadata.append(metadata)
    final_metadata = pd.concat(final_metadata, ignore_index=True)
    print("{} outdoor anafi videos".format(videos_summary["anafi"]["outdoor"]))
    print("{} indoor anafi videos".format(videos_summary["anafi"]["indoor"]))
    print("{} generic videos".format(videos_summary["generic"]))

    print("{} frames in total".format(len(final_metadata)))

    cam_fields = ["width", "height", "framerate", "picture_hfov", "picture_vfov", "camera_model"]
    cameras_dataframe = final_metadata[final_metadata["model"] == "anafi"][cam_fields].drop_duplicates()
    cameras_dataframe = register_new_cameras(cameras_dataframe, database, colmap_cameras)
    final_metadata["camera_id"] = 0
    for cam_id, row in cameras_dataframe.iterrows():
        final_metadata.loc[(final_metadata[cam_fields] == row).all(axis=1), "camera_id"] = cam_id
    if any(final_metadata["model"] == "generic"):
        print("Undefined remaining cameras, assigning generic models to them")
        generic_frames = final_metadata[final_metadata["model"] == "generic"]
        generic_cam_fields = cam_fields + ["video"]
        generic_cameras_dataframe = generic_frames[generic_cam_fields]
        fixed_camera = True
        if fixed_camera:
            generic_cameras_dataframe = generic_cameras_dataframe.drop_duplicates()
        generic_cameras_dataframe = register_new_cameras(generic_cameras_dataframe, database, colmap_cameras)
        if fixed_camera:
            for cam_id, row in generic_cameras_dataframe.iterrows():
                final_metadata.loc[(final_metadata[generic_cam_fields] == row).all(axis=1), "camera_id"] = cam_id
        else:
            final_metadata.loc[generic_frames.index, "camera_id"] = generic_cameras_dataframe.index
        cameras_dataframe = cameras_dataframe.append(generic_cameras_dataframe)
    print("Cameras : ")
    print(cameras_dataframe)

    to_extract = total_frames - len(existing_pictures) - sum(final_metadata["sampled"])

    if to_extract <= 0:
        pass
    elif to_extract < len(final_metadata):
        print("subsampling based on K-Means, to get {}"
              " frames from videos, for a total of {} frames".format(to_extract, total_frames))
        final_metadata = optimal_sample(final_metadata, total_frames - len(existing_pictures),
                                        orientation_weight,
                                        resolution_weight)
        print("Done.")
    else:
        final_metadata["sampled"] = True

    print("Constructing COLMAP model with {:,} frames".format(sum(final_metadata["sampled"])))

    database.commit()
    thorough_db.copy(tempfile_database)
    temp_database = db.COLMAPDatabase.connect(tempfile_database)

    final_metadata["image_path"] = ""
    final_metadata["db_id"] = -1
    for current_id, row in tqdm(final_metadata.iterrows(), total=len(final_metadata)):
        video = row["video"]
        frame = row["frame"]
        camera_id = row["camera_id"]
        current_image_path = video_output_folders[video].relpath(image_path) / video.stem + "_{:05d}.jpg".format(frame)

        final_metadata.at[current_id, "image_path"] = current_image_path
        db_image_id = temp_database.add_image(current_image_path, int(camera_id))
        final_metadata.at[current_id, "db_id"] = db_image_id

        if row["sampled"]:
            frame_qvec = row[["frame_quat_w",
                              "frame_quat_x",
                              "frame_quat_y",
                              "frame_quat_z"]].values
            if True in pd.isnull(frame_qvec):
                frame_qvec = np.array([1, 0, 0, 0])
            x, y, z = row[["x", "y", "z"]]
            frame_tvec = np.array([x, y, z])
            if row["location_valid"]:
                frame_gps = row[["location_longitude", "location_latitude", "location_altitude"]]
            else:
                frame_gps = np.full(3, np.NaN)

            world_qvec, world_tvec = world_coord_from_frame(frame_qvec, frame_tvec)
            database.add_image(current_image_path, int(camera_id), prior_t=frame_gps, image_id=db_image_id)
            images[db_image_id] = rm.Image(id=db_image_id, qvec=world_qvec, tvec=world_tvec,
                                           camera_id=camera_id, name=current_image_path,
                                           xys=[], point3D_ids=[])

    database.commit()
    database.close()
    temp_database.commit()
    temp_database.close()
    rm.write_model(colmap_cameras, images, {}, output_video_folder, "." + output_colmap_format)
    print("COLMAP model created")

    thorough_georef, thorough_paths = get_georef(final_metadata[final_metadata["sampled"]])
    path_lists_output["thorough"] = {}
    path_lists_output["thorough"]["frames"] = thorough_paths
    path_lists_output["thorough"]["georef"] = thorough_georef

    print("Extracting frames from videos")

    for v in tqdm(videos_list):
        video_metadata = final_metadata[final_metadata["video"] == v]
        by_time = video_metadata.set_index(pd.to_datetime(video_metadata["time"], unit="us"))
        video_folder = video_output_folders[v]
        video_metadata.to_csv(video_folder/"metadata.csv")
        path_lists_output[v] = {}
        video_metadata_1fps = by_time.resample("{:.3f}S".format(1/fps)).first()
        georef, frame_paths = get_georef(video_metadata_1fps)
        path_lists_output[v]["frames_lowfps"] = frame_paths
        path_lists_output[v]["georef_lowfps"] = georef
        num_chunks = len(video_metadata) // max_sequence_length + 1
        chunks = [list(frames) for frames in np.array_split(video_metadata["image_path"],
                                                            num_chunks)]
        # Add some overlap between chunks, in order to ease the model merging afterwards
        for chunk, next_chunk in zip(chunks, chunks[1:]):
            chunk.extend(next_chunk[:10])
        path_lists_output[v]["frames_full"] = chunks

        if save_space:
            frame_ids = set(video_metadata[video_metadata["sampled"]]["frame"].values) | \
                set(video_metadata_1fps["frame"].values)
            frame_ids = sorted(list(frame_ids))
            if len(frame_ids) > 0:
                extracted_frames = env["ffmpeg"].extract_specific_frames(v, video_folder, frame_ids)
        else:
            extracted_frames = env["ffmpeg"].extract_images(v, video_folder)
        set_gps(extracted_frames, video_metadata, image_path)

    return path_lists_output, video_output_folders


if __name__ == '__main__':
    args = parser.parse_args()
    env = vars(args)
    env["videos_list"] = sum((list(args.video_folder.walkfiles('*{}'.format(ext))) for ext in args.vid_ext), [])
    output_video_folder = args.colmap_img_root / "Videos"
    output_video_folder.makedirs_p()
    env["image_path"] = args.colmap_img_root
    env["output_video_folder"] = output_video_folder
    env["existing_pictures"] = sum((list(args.colmap_img_root.walkfiles('*{}'.format(ext))) for ext in args.pic_ext), [])
    env["pdraw"] = PDraw(args.nw, verbose=args.verbose)
    env["ffmpeg"] = FFMpeg(verbose=args.verbose)
    env["output_colmap_format"] = args.output_format

    if args.centroid_path is not None:
        centroid = np.loadtxt(args.centroid_path)
    else:
        centroid = np.zeros(3)
    env["centroid"] = centroid
    lists, extracted_video_folders = process_video_folder(**env)

    if lists is not None:
        with open(args.colmap_img_root/"video_frames_for_thorough_scan.txt", "w") as f:
            f.write("\n".join(lists["thorough"]["frames"]) + "\n")
        with open(args.colmap_img_root/"georef.txt", "w") as f:
            f.write("\n".join(lists["thorough"]["georef"]))
        for v in env["videos_list"]:
            video_folder = extracted_video_folders[v]
            with open(video_folder / "lowfps.txt", "w") as f:
                f.write("\n".join(lists[v]["frames_lowfps"]) + "\n")
            with open(video_folder / "georef.txt", "w") as f:
                f.write("\n".join(lists["thorough"]["georef"]) + "\n")
                f.write("\n".join(lists[v]["georef_lowfps"]) + "\n")
            for j, l in enumerate(lists[v]["frames_full"]):
                with open(video_folder / "full_chunk_{}.txt".format(j), "w") as f:
                    f.write("\n".join(l) + "\n")
