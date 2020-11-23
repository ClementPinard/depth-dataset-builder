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
                                    'location in a COLMAP file for visualization',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--video_folder', metavar='DIR',
                    help='path to videos', type=Path)
parser.add_argument('--system', default='epsg:2154',
                    help='coordinates system used for GPS, should be the same as the LAS files used')
parser.add_argument('--centroid_path', default=None, help="path to centroid generated in las2ply.py")
parser.add_argument('--colmap_img_root', metavar='DIR', type=Path,
                    help="folder that will be used as \"image_path\" parameter when using COLMAP", required=True)
parser.add_argument('--output_format', metavar='EXT', default="bin", choices=["bin", "txt"],
                    help='format of the COLMAP file that will be outputed, used for visualization only')
parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"],
                    help="format of video files that will be scraped from input folder")
parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"],
                    help='format of images that will be scraped from already existing images in colmap image_path folder')
parser.add_argument('--nw', default='',
                    help="native-wrapper.sh file location (see Anafi SDK documentation)")
parser.add_argument('--fps', default=1, type=int,
                    help="framerate at which videos will be scanned WITH reconstruction")
parser.add_argument('--total_frames', default=200, type=int, help="number of frames used for thorough photogrammetry")
parser.add_argument('--max_sequence_length', default=1000, help='Number max of frames for a chunk. '
                    'This is for RAM purpose, as loading feature matches of thousands of frames can take up GBs of RAM')
parser.add_argument('--orientation_weight', default=1, type=float,
                    help="Weight applied to orientation during optimal sample. "
                    "Higher means two pictures with same location but different orientation will be considered farer apart")
parser.add_argument('--resolution_weight', default=1, type=float, help="same as orientation, but with image size")
parser.add_argument('--save_space', action="store_true",
                    help="if selected, will only extract from ffmpeg frames used for thorough photogrammetry")
parser.add_argument('--thorough_db', type=Path, help="output db file which will be used by COLMAP for photogrammetry")
parser.add_argument('--generic_model', default='OPENCV',
                    help='COLMAP model for generic videos. Same zoom level assumed throughout the whole video. '
                    'See https://colmap.github.io/cameras.html')
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
            if row["location_valid"] and not row['indoor']:
                set_gps_location(frame,
                                 lat=row["location_latitude"],
                                 lng=row["location_longitude"],
                                 altitude=row["location_altitude"])


def get_georef(metadata):
    relevant_data = metadata[["location_valid", "image_path", "x", "y", "z"]]
    path_list = []
    georef_list = []
    for _, (loc_valid, path, x, y, alt) in relevant_data.iterrows():
        path_list.append(path)
        if loc_valid:
            georef_list.append("{} {} {} {}\n".format(path, x, y, alt))
    return georef_list, path_list


def optimal_sample(metadata, num_frames, orientation_weight, resolution_weight):
    # already sampled frames are discarded as we want to sample frames in addition to them
    valid_metadata = metadata[~metadata["sampled"]].dropna()
    if len(valid_metadata) == 0:
        return metadata
    XYZ = valid_metadata[["x", "y", "z"]].values
    axis_angle = valid_metadata[["frame_quat_x", "frame_quat_y", "frame_quat_z"]].values
    if "indoor" in valid_metadata.keys() and (True in valid_metadata["indoor"].unique()):
        # We have indoor videos, without absolute positions. We assume each video is very far
        # from the other ones. As such we will have an optimal subsampling of each video
        # It won't leverage video proximity from each other but it's better than nothing
        diameter = (XYZ.max(axis=0) - XYZ.min(axis=0))
        indoor_videos = valid_metadata.loc[valid_metadata["indoor"]]["video"].unique()
        new_centroids = 2 * diameter * np.linspace(0, 10, len(indoor_videos)).reshape(-1, 1)
        for centroid, v in zip(new_centroids, indoor_videos):
            video_index = (valid_metadata["video"] == v).values
            XYZ[video_index] += centroid

    weighted_point_cloud = np.concatenate([XYZ, orientation_weight * axis_angle], axis=1)

    if resolution_weight == 0:
        weights = None
    else:
        frame_size = valid_metadata["video_quality"].values
        weights = frame_size ** resolution_weight
    km = KMeans(n_clusters=num_frames).fit(weighted_point_cloud, sample_weight=weights)
    closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, weighted_point_cloud)
    metadata.at[valid_metadata.index[closest], "sampled"] = True
    return metadata


def register_new_cameras(metadata, device, fields, database, camera_dict):
    camera_ids = []
    cameras_dataframe = metadata[metadata["device"] == device][["device"] + fields].drop_duplicates()
    for _, row in cameras_dataframe.iterrows():
        camera_model, w, h, params = row.reindex(["camera_model", "width", "height", "camera_params"])
        model_id = rm.CAMERA_MODEL_NAMES[camera_model].model_id
        num_params = rm.CAMERA_MODEL_NAMES[camera_model].num_params
        assert num_params >= len(params), "Got {} params for camera {}".format(len(params), camera_model)
        # Single focal models are SIMPLE_PINHOLE, SIMPLE_RADIAL, SIMPLE_RADIAL_FISHEYE, RADIAL and RADIAL_FISHEYE
        single_focal = ('SIMPLE' in camera_model) or ('RADIAL' in camera_model)
        num_focals = 1 if single_focal else 2
        params = np.array(list(params) + [0] * (num_params - len(params)))

        # prior_focal_length is whether or not COLMAP should rely on it.
        prior_focal_length = all(params[:num_focals] != 0)
        # For unknown focal_length, put a generic placeholder
        params[:num_focals][params[:num_focals] == 0] = w / 2
        # We can get less params than actual params if they are unknown. We then pad it with zeros
        db_id = database.add_camera(model_id, int(w), int(h), params, prior_focal_length=prior_focal_length)
        camera_ids.append(db_id)
        camera_dict[db_id] = rm.Camera(id=db_id,
                                       model=camera_model,
                                       width=int(w),
                                       height=int(h),
                                       params=params)
        metadata.loc[(metadata[["device"] + fields] == row).all(axis=1), "camera_id"] = db_id
    ids_series = pd.Series(camera_ids)
    return cameras_dataframe.set_index(ids_series)


def get_video_metadata(v, output_video_folder, system, generic_model='OPENCV', ** env):
    width, height, framerate, num_frames = env["ffmpeg"].get_size_and_framerate(v)
    video_output_folder = output_video_folder / "{}x{}".format(width, height) / v.stem

    def string_to_tuple(tuple_string):
        assert(tuple_string[0] == '(' and tuple_string[-1] == ')')
        return tuple([float(f) for f in tuple_string[1:-1].split(', ')])

    def generic_metadata():
        metadata = pd.DataFrame({"video": [v] * num_frames})
        metadata["height"] = height
        metadata["width"] = width
        metadata["framerate"] = framerate
        metadata["video_quality"] = height * width / framerate
        metadata['frame'] = metadata.index + 1
        # timestemp is in microseconds
        metadata['time'] = 1e6 * metadata.index / framerate
        metadata['indoor'] = True
        metadata['location_valid'] = False
        metadata["device"] = "generic"
        metadata["camera_model"] = generic_model
        metadata["frame_quat_w"] = np.NaN
        metadata["frame_quat_x"] = np.NaN
        metadata["frame_quat_y"] = np.NaN
        metadata["frame_quat_z"] = np.NaN
        metadata["x"] = np.NaN
        metadata["y"] = np.NaN
        metadata["z"] = np.NaN
        metadata["camera_params"] = [tuple()] * len(metadata)
        return metadata

    # First, try to open the CSV file {video name}_metadata.csv which should contain the metadata
    # If it fails, try to get metadata from MP4 by using PDraw
    # At last resort, simply assume generic parameters

    metadata_file_path = v.parent / "{}_metadata.csv".format(v.stem)
    if metadata_file_path.isfile():
        metadata = pd.read_csv(metadata_file_path)
        # check that the pandas dataframe is well formed
        keys_to_check = ["camera_model", "camera_params", "x", "y", "z",
                         "frame_quat_w", "frame_quat_x", "frame_quat_y", "frame_quat_z",
                         "location_valid", "time"]
        for k in keys_to_check:
            assert k in metadata.keys(), "Metadata file does not contain required field {}".format(k)
        metadata["camera_params"] = metadata["camera_params"].apply(string_to_tuple)
        if "frame" not in metadata.keys():
            metadata["frame"] = range(1, len(metadata) + 1)
        metadata['video'] = v
        if 'indoor' not in metadata.keys():
            metadata['indoor'] = len(metadata[metadata["location_valid"]]) > 0
        if 'video_quality' not in metadata.keys():
            metadata["video_quality"] = height * width / framerate
        device = "other"
    else:
        try:
            proj = Proj(system)
            metadata = am.extract_metadata(v.parent, v, env["pdraw"], proj,
                                           width, height, framerate)
            metadata["camera_model"] = "PINHOLE"
            device = "anafi"
        except Exception:
            # No metadata found, construct a simpler dataframe without location
            metadata = generic_metadata()
            device = "generic"
    metadata["num_frames"] = num_frames
    metadata["device"] = device
    return metadata, device, video_output_folder


def process_video_folder(videos_list, existing_pictures, output_video_folder, image_path, centroid,
                         thorough_db, fps=1, total_frames=500, orientation_weight=1, resolution_weight=1,
                         output_colmap_format="bin", save_space=False, include_lowfps_thorough=False,
                         max_sequence_length=1000, num_neighbours=10, existing_georef=False, **env):
    metadata_list = []
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
    videos_summary = {"anafi": {"indoor": 0, "outdoor": 0},
                      "other": {"indoor": 0, "outdoor": 0},
                      "generic": 0}
    for v in tqdm(videos_list):
        metadata, device, output_folder = get_video_metadata(v, output_video_folder, **env)
        video_output_folders[v] = output_folder
        output_folder.makedirs_p()

        if include_lowfps_thorough:
            by_time = metadata.set_index(pd.to_datetime(metadata["time"], unit="us"))
            by_time_lowfps = by_time.resample("{:.3f}S".format(1/fps)).first()
            metadata["sampled"] = by_time["time"].isin(by_time_lowfps["time"]).values
        else:
            metadata["sampled"] = False
        if device == "generic":
            videos_summary["generic"] += 1
        else:
            raw_positions = metadata[["x", "y", "z"]]
            if metadata["indoor"].iloc[0]:
                videos_summary[device]["indoor"] += 1
            else:
                videos_summary[device]["outdoor"] += 1
            if sum(metadata["location_valid"]) > 0:
                if centroid is None:
                    '''No centroid (possibly because there was no georeferenced lidar pointcloud in the first place)
                    set it as the first valid GPS position of the first outdoor video'''
                    centroid = raw_positions[metadata["location_valid"]].iloc[0].values
                zero_centered_positions = raw_positions.values - centroid
                radius = np.max(np.abs(zero_centered_positions))
                if radius > 1000:
                    print("Warning, your positions coordinates are most likely too high, have you configured the right GPS system ?")
                    print("It should be the same as the one used for the Lidar point cloud")
                metadata["x"], metadata["y"], metadata["z"] = zero_centered_positions.transpose()
        metadata_list.append(metadata)
    final_metadata = pd.concat(metadata_list, ignore_index=True)
    print("{} outdoor anafi videos".format(videos_summary["anafi"]["outdoor"]))
    print("{} indoor anafi videos".format(videos_summary["anafi"]["indoor"]))
    print("{} indoor other videos".format(videos_summary["other"]["outdoor"]))
    print("{} indoor other videos".format(videos_summary["other"]["indoor"]))
    print("{} generic videos".format(videos_summary["generic"]))

    if((not existing_georef) and (sum(final_metadata["location_valid"]) == 0) and (videos_summary["anafi"]["indoor"] > 0)):
        # We have no GPS data but we have navdata, which will help rescale the colmap model
        # Take the longest video and do as if the GPS was valid
        indoor_video_diameters = {}
        for md in metadata_list:
            if (metadata["device"].iloc[0] != "anafi") or (not metadata["indoor"].iloc[0]):
                continue
            positions = md[["x", "y", "z"]].values
            video_displacement_diameter = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
            if not np.isnan(video_displacement_diameter):
                indoor_video_diameters[video_displacement_diameter] = v

        if len(indoor_video_diameters) > 0:
            longest_video = indoor_video_diameters[max(indoor_video_diameters)]
            print("Only indoor videos used, will use {} for COLMAP rescaling".format(longest_video))
            video_index = final_metadata["video"] == longest_video
            final_metadata.loc[video_index, "location_valid"] = True

    print("{} frames in total".format(len(final_metadata)))

    final_metadata["camera_id"] = 0
    # Set up Anafi cameras, zoom included
    cam_fields = ["camera_model", "width", "height", "camera_params"]
    cam_dfs = []

    if any(final_metadata["device"] == "other"):
        cam_dfs.append(register_new_cameras(final_metadata, "other", cam_fields, database, colmap_cameras))
    if any(final_metadata["device"] == "anafi"):
        # For anafi we don't treat cameras the same if the framerate is different
        # because potentially different rectification algorithms are applied
        anafi_cam_fields = cam_fields + ["framerate"]
        cam_dfs.append(register_new_cameras(final_metadata, "anafi", anafi_cam_fields, database, colmap_cameras))
    if any(final_metadata["device"] == "generic"):
        print("Undefined remaining devices, assigning generic models to them")
        # Fix a single camera per video. This doesn't support different levels of zoom, but
        # COLMAP is not robust to too many different independant camera models
        generic_cam_fields = cam_fields + ["video"]
        cam_dfs.append(register_new_cameras(final_metadata, "generic", generic_cam_fields, database, colmap_cameras))
    print("Cameras : ")
    print(pd.concat(cam_dfs))

    to_extract = total_frames - len(existing_pictures) - sum(final_metadata["sampled"])

    if to_extract <= 0:
        pass
    elif to_extract < len(final_metadata):
        print("subsampling based on K-Means, to get {}"
              " frames from videos, for a total of {} frames".format(to_extract, total_frames))
        final_metadata = optimal_sample(final_metadata, to_extract,
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
            if row["location_valid"] and not row['indoor']:
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
            chunk.extend(next_chunk[:num_neighbours])
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
