from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from colmap_util.read_model import read_images_text
from pyproj import Proj
from edit_exif import get_gps_location
import meshlab_xml_writer as mxw
import add_video_to_db as avtd
import extract_video_from_model as evfm
from path import Path
import numpy as np
import videos_to_colmap as v2c
import viz_depth as vd
import generate_sky_masks as gsm
import pandas as pd
import las2ply
import rawpy
import imageio
import tempfile

global_steps = ["Point Cloud Preparation",
                "Pictures preparation",
                "Extracting Videos and selecting optimal frames for a thorough scan",
                "First thorough photogrammetry",
                "Alignment of photogrammetric reconstruction with Lidar point cloud",
                "Occlusion Mesh computing",
                "Video localization",
                "Ground Truth creation"]

per_vid_steps_1 = ["Full video extraction",
                   "Sky mask generation",
                   "Complete photogrammetry with video at 1 fps",
                   "Localizing remaining frames",
                   "Re-Alignment of triangulated points with Lidar point cloud"]
per_vid_steps_2 = ["Creating Ground truth data",
                   "Create video with GT vizualisation",
                   "Convert to KITTI format"]

parser = ArgumentParser(description='Main pipeline, from LIDAR pictures and videos to GT depth enabled videos',
                        formatter_class=ArgumentDefaultsHelpFormatter)

main_parser = parser.add_argument_group("Main options")
main_parser.add_argument('--input_folder', metavar='PATH', default=Path("."), type=Path,
                         help="Folder with LAS point cloud, videos, and images")
main_parser.add_argument('--workspace', metavar='PATH', default=Path("."),
                         help='path to workspace where COLMAP operations will be done', type=Path)
main_parser.add_argument('--output_folder', metavar='PATH', default=Path("."),
                         help='path to output folder : must be big !', type=Path)
main_parser.add_argument('--skip_step', metavar="N", nargs="*", default=[], type=int)
main_parser.add_argument('--begin_step', metavar="N", type=int, default=None)
main_parser.add_argument('--show_steps', action="store_true")
main_parser.add_argument('-v', '--verbose', action="count", default=0)
main_parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"])
main_parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"])
main_parser.add_argument('--raw_ext', nargs='+', default=[".ARW", ".NEF", ".DNG"])
main_parser.add_argument('--dense', action="store_true")
main_parser.add_argument('--fine_sift_features', action="store_true")
main_parser.add_argument('--triangulate', action="store_true")
main_parser.add_argument('--save_space', action="store_true")
main_parser.add_argument('--add_new_videos', action="store_true")
main_parser.add_argument('--resume_work', action="store_true")

pcp_parser = parser.add_argument_group("PointCLoud preparation")
pcp_parser.add_argument("--pointcloud_resolution", default=0.1, type=float)
pcp_parser.add_argument("--SOR", default=[10, 6], nargs=2, type=int)

ve_parser = parser.add_argument_group("Video extractor")
ve_parser.add_argument('--total_frames', default=500, type=int)
ve_parser.add_argument('--orientation_weight', default=1, type=float)
ve_parser.add_argument('--resolution_weight', default=1, type=float)
ve_parser.add_argument('--num_neighbours', default=10, type=int)
ve_parser.add_argument('--system', default="epsg:2154")
ve_parser.add_argument('--lowfps', default=1, type=int)
ve_parser.add_argument('--max_sequence_length', default=4000, type=int)

exec_parser = parser.add_argument_group("Executable files")
exec_parser.add_argument('--log', default=None, type=Path)
exec_parser.add_argument('--nw', default="native-wrapper.sh", type=Path,
                         help="native-wrapper.sh file location")
exec_parser.add_argument("--colmap", default="colmap", type=Path,
                         help="colmap exec file location")
exec_parser.add_argument("--eth3d", default="../dataset-pipeline/build",
                         type=Path, help="ETH3D detaset pipeline exec files folder location")
exec_parser.add_argument("--ffmpeg", default="ffmpeg", type=Path)
exec_parser.add_argument("--pcl_util", default="pcl_util/build", type=Path)

vr_parser = parser.add_argument_group("Video Registration")
vr_parser.add_argument("--vocab_tree", type=Path, default="vocab_tree_flickr100K_words256K.bin")

om_parser = parser.add_argument_group("Occlusion Mesh")
om_parser.add_argument('--normal_radius', default=0.2, type=float)
om_parser.add_argument('--mesh_resolution', default=0.2, type=float)
om_parser.add_argument('--splat_threshold', default=0.1, type=float)


def print_workflow():
    print("Global steps :")
    for i, s in enumerate(global_steps):
        print("{}:\t{}".format(i, s))

    print("Per video steps :")
    for i, s in enumerate(per_vid_steps_1):
        print("\t{}:\t{}".format(i, s))
    for i, s in enumerate(per_vid_steps_2):
        print("\t{}:\t{}".format(i, s))


def print_step(step_number, step_name):
        print("\n\n=================")
        print("Step {}".format(step_number))
        print(step_name)
        print("=================")


def prepare_point_clouds(pointclouds, lidar_path, verbose, eth3d, pcl_util, SOR, pointcloud_resolution, save_space, **env):
    converted_clouds = []
    output_centroid = None
    for pc in pointclouds:
        ply, centroid = las2ply.load_and_convert(input_file=pc,
                                                 output_folder=lidar_path,
                                                 verbose=verbose >= 1)
        if pc.ext[1:].upper() == "LAS":
            if output_centroid is None:
                output_centroid = centroid
        pcl_util.filter_cloud(input_file=ply, output_file=ply.stripext() + "_filtered.ply", knn=SOR[0], std=SOR[1])
        pcl_util.subsample(input_file=ply.stripext() + "_filtered.ply", output_file=ply.stripext() + "_subsampled.ply", resolution=pointcloud_resolution)

        converted_clouds.append(ply.stripext() + "_subsampled.ply")
    temp_mlp = env["workspace"] / "lidar_unaligned.mlp"
    mxw.create_project(temp_mlp, converted_clouds, labels=None, transforms=None)
    if len(converted_clouds) > 1:
        eth3d.align_with_ICP(temp_mlp, env["lidar_mlp"], scales=5)
    else:
        temp_mlp.move(env["lidar_mlp"])

    return converted_clouds, output_centroid


def extract_gps_and_path(existing_pictures, image_path, system, centroid, **env):
    proj = Proj(system)
    georef_list = []
    for img in existing_pictures:
        gps = get_gps_location(img)
        if gps is not None:
            lat, lng, alt = gps
            x, y = proj(lng, lat)
            x -= centroid[0]
            y -= centroid[1]
            alt -= centroid[2]
            georef_list.append("{} {} {} {}\n".format(img.relpath(image_path), x, y, alt))
    return georef_list


def extract_pictures_to_workspace(input_folder, image_path, workspace, colmap, raw_ext, pic_ext, fine_sift_features, **env):
    picture_folder = input_folder / "Pictures"
    picture_folder.merge_tree(image_path)
    raw_files = sum((list(image_path.walkfiles('*{}'.format(ext))) for ext in raw_ext), [])
    for raw in raw_files:
        if not any((raw.stripext() + ext).isfile() for ext in pic_ext):
            raw_array = rawpy.imread(raw)
            rgb = raw_array.postprocess()
            imageio.imsave(raw.stripext() + ".jpg", rgb)
        raw.remove()
    gsm.process_folder(folder_to_process=image_path, image_path=image_path, pic_ext=pic_ext, **env)
    colmap.extract_features(per_sub_folder=True, fine=fine_sift_features)
    return sum((list(image_path.walkfiles('*{}'.format(ext))) for ext in pic_ext), [])


def extract_videos_to_workspace(video_path, **env):
    return v2c.process_video_folder(output_video_folder=video_path, **env)


def check_input_folder(path):
    def print_error_string():
        print("Error, bad input folder structure")
        print("Expected :")
        print(str(path/"Lidar"))
        print(str(path/"Pictures"))
        print(str(path/"Videos"))
        print()
        print("but got :")
        print("\n".join(str(d) for d in path.dirs()))

    if all((path/d).isdir() for d in ["Lidar", "Pictures", "Videos"]):
        return
    else:
        print_error_string()


def prepare_workspace(path, env):
    for dirname, key in zip(["Lidar", "Pictures", "Masks",
                             "Pictures/Videos", "Thorough/0", "Thorough/georef",
                             "Videos_reconstructions"],
                            ["lidar_path", "image_path", "mask_path",
                             "video_path", "thorough_recon", "georef_recon",
                             "video_recon"]):
        (path/dirname).makedirs_p()
        env[key] = path/dirname

    env["video_frame_list_thorough"] = env["image_path"] / "video_frames_for_thorough_scan.txt"
    env["georef_frames_list"] = env["image_path"] / "georef.txt"

    env["lidar_mlp"] = env["workspace"] / "lidar.mlp"
    env["lidar_ply"] = env["lidar_path"] / "aligned.ply"
    env["aligned_mlp"] = env["workspace"] / "aligned_model.mlp"
    env["occlusion_ply"] = env["lidar_path"] / "occlusion_model.ply"
    env["splats_ply"] = env["lidar_path"] / "splats_model.ply"
    env["occlusion_mlp"] = env["lidar_path"] / "occlusions.mlp"
    env["splats_mlp"] = env["lidar_path"] / "splats.mlp"
    env["georefrecon_ply"] = env["georef_recon"] / "georef_reconstruction.ply"
    env["matrix_path"] = env["workspace"] / "matrix_thorough.txt"
    env["indexed_vocab_tree"] = env["workspace"] / "vocab_tree_thorough.bin"


def main():
    args = parser.parse_args()
    env = vars(args)
    if args.show_steps:
        print_workflow()
    if args.add_new_videos:
        args.skip_step += [1, 2, 4, 5, 6]
    if args.begin_step is not None:
        args.skip_step += list(range(args.begin_step))
    check_input_folder(args.input_folder)
    prepare_workspace(args.workspace, env)
    colmap = Colmap(db=args.workspace/"thorough_scan.db",
                    image_path=env["image_path"],
                    mask_path=env["mask_path"],
                    binary=args.colmap,
                    quiet=args.verbose < 1,
                    logfile=args.log)
    env["colmap"] = colmap
    ffmpeg = FFMpeg(args.ffmpeg, quiet=args.verbose < 2, logfile=args.log)
    env["ffmpeg"] = ffmpeg
    pdraw = PDraw(args.nw, quiet=args.verbose < 2, logfile=args.log)
    env["pdraw"] = pdraw
    eth3d = ETH3D(args.eth3d, env["image_path"], quiet=args.verbose < 1, logfile=args.log)
    env["eth3d"] = eth3d
    pcl_util = PCLUtil(args.pcl_util, quiet=args.verbose < 2, logfile=args.log)
    env["pcl_util"] = pcl_util

    las_files = (args.input_folder/"Lidar").files("*.las")
    ply_files = (args.input_folder/"Lidar").files("*.ply")
    input_pointclouds = las_files + ply_files
    env["videos_list"] = sum((list((args.input_folder/"Videos").walkfiles('*{}'.format(ext))) for ext in args.vid_ext), [])

    i = 1
    if i not in args.skip_step:
        print_step(i, "Point Cloud Preparation")
        env["pointclouds"], env["centroid"] = prepare_point_clouds(input_pointclouds, **env)
    else:
        env["pointclouds"] = env["lidar_path"].files("*inliers.ply")
        centroid_path = sorted(env["lidar_path"].files("*_centroid.txt"))[0]
        env["centroid"] = np.loadtxt(centroid_path)

    i += 1
    if i not in args.skip_step:
        print_step(i, "Pictures preparation")
        env["existing_pictures"] = extract_pictures_to_workspace(**env)
    else:
        env["existing_pictures"] = sum((list(env["image_path"].walkfiles('*{}'.format(ext))) for ext in env["pic_ext"]), [])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Extracting Videos and selecting optimal frames for a thorough scan")
        existing_georef = extract_gps_and_path(**env)
        path_lists, env["videos_output_folders"] = extract_videos_to_workspace(fps=args.lowfps, **env)
        if path_lists is not None:
            with open(env["video_frame_list_thorough"], "w") as f:
                f.write("\n".join(path_lists["thorough"]["frames"]))
            with open(env["georef_frames_list"], "w") as f:
                f.write("\n".join(existing_georef) + "\n")
                f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
            for v in env["videos_list"]:
                video_folder = env["videos_output_folders"][v]
                with open(video_folder / "to_scan.txt", "w") as f:
                    f.write("\n".join(path_lists[v]["frames_lowfps"]) + "\n")
                with open(video_folder / "georef.txt", "w") as f:
                    f.write("\n".join(existing_georef) + "\n")
                    f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
                    f.write("\n".join(path_lists[v]["georef_lowfps"]) + "\n")
                for j, l in enumerate(path_lists[v]["frames_full"]):
                    with open(video_folder / "full_{}.txt".format(i), "w") as f:
                        f.write("\n".join(l) + "\n")
    else:
        env["videos_output_folders"] = {}
        by_name = {v.namebase: v for v in env["videos_list"]}
        for folder in env["video_path"].walkdirs():
            video_name = folder.basename()
            if video_name in by_name.keys():
                env["videos_output_folders"][by_name[video_name]] = folder

    i += 1
    if i not in args.skip_step:
        print_step(i, "First thorough photogrammetry")
        gsm.process_folder(folder_to_process=env["video_path"], **env)
        colmap.extract_features(image_list=env["video_frame_list_thorough"], fine=args.fine_sift_features)
        colmap.index_images(vocab_tree_output=env["indexed_vocab_tree"], vocab_tree_input=args.vocab_tree)
        colmap.match()
        colmap.map(output=env["thorough_recon"].parent)

    i += 1
    if i not in args.skip_step:
        print_step(i, "Alignment of photogrammetric reconstruction with Lidar point cloud")

        colmap.align_model(output=env["georef_recon"],
                           input=env["thorough_recon"],
                           ref_images=env["georef_frames_list"])
        if args.dense:
            print_step("{} (bis)".format(i), "Point cloud densificitation")
            dense_workspace = env["thorough_recon"]/"dense"
            colmap.undistort(input=env["georef_recon"], output=dense_workspace)
            colmap.dense_stereo(workspace=dense_workspace)
            colmap.stereo_fusion(workspace=dense_workspace, output=env["georefrecon_ply"])
        else:
            colmap.export_model(output=env["georefrecon_ply"],
                                input=env["georef_recon"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Occlusion Mesh computing")

        with_normals_path = env["lidar_path"] / "with_normals.ply"
        eth3d.compute_normals(with_normals_path, env["lidar_mlp"], neighbor_radius=args.normal_radius)
        pcl_util.triangulate_mesh(env["occlusion_ply"], with_normals_path, resolution=args.mesh_resolution)
        eth3d.create_splats(env["splats_ply"], with_normals_path, env["occlusion_ply"], threshold=args.splat_threshold)
        pcl_util.register_reconstruction(georef=env["georefrecon_ply"],
                                         lidar=with_normals_path,
                                         output_matrix=env["matrix_path"],
                                         max_distance=10)
    if env["matrix_path"].isfile():
        matrix = np.linalg.inv(np.fromfile(env["matrix_path"], sep=" ").reshape(4, 4))
    else:
        print("Error, no registration matrix can be found")
        matrix = np.eye(4)

    mxw.apply_transform_to_project(env["lidar_mlp"], env["aligned_mlp"], matrix)
    mxw.create_project(env["occlusion_mlp"], [env["occlusion_ply"]], transforms=[matrix])
    mxw.create_project(env["splats_mlp"], [env["splats_ply"]], transforms=[matrix])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Video localization")
        for j, v in enumerate(env["videos_list"]):
            print(v)
            current_video_folder = env["videos_output_folders"][v]
            thorough_db = args.workspace / "thorough_scan.db"
            lowfps_db = current_video_folder / "video_low_fps.db"
            current_metadata = current_video_folder / "metadata.csv"
            map_image_list_path = current_video_folder / "to_scan.txt"
            full_image_list_path = current_video_folder.files("full*.txt")
            full_dbs = [current_video_folder / fp.namebase + ".db" for fp in full_image_list_path]
            video_output_model = env["video_recon"] / v.namebase
            chunk_output_models = [video_output_model / "chunk_{}".format(index) for index in range(len(full_image_list_path))]
            final_output_model = video_output_model / "final"
            # Perform checks if it has not already been computed
            if args.resume_work and final_output_model.isdir():
                print("already done")
                continue

            i_pv = 1
            print("Now working on video {} [{}/{}]".format(v, j + 1, len(env["videos_list"])))
            thorough_db.copy(lowfps_db)
            colmap.db = lowfps_db

            print_step(i_pv, "Full video extraction")
            if args.save_space:
                existing_images = list(current_video_folder.files())
                ffmpeg.extract_images(v, current_video_folder)
            else:
                print("Already Done.")

            i_pv += 1
            print_step(i_pv, "Sky mask generation")
            gsm.process_folder(folder_to_process=current_video_folder, **env)

            i_pv += 1
            print_step(i_pv, "Complete photogrammetry with video at {} fps".format(args.lowfps))
            avtd.add_to_db(lowfps_db, current_metadata, map_image_list_path)
            colmap.extract_features(image_list=map_image_list_path, fine=args.fine_sift_features)
            colmap.match(method="sequential", vocab_tree=env["indexed_vocab_tree"])
            video_output_model.makedirs_p()
            colmap.map(output=video_output_model, input=env["georef_recon"])
            # when colmap map is called, the model is normalized so we have georegister it again
            colmap.align_model(output=video_output_model,
                               input=video_output_model,
                               ref_images=env["georef_frames_list"])

            i_pv += 1
            print_step(i_pv, "Localizing remaining frames")
            # get image_ids in a fake database with all the chunks
            frame_ids_per_chunk = []
            temp_db = Path(tempfile.NamedTemporaryFile().name)
            lowfps_db.copy(temp_db)
            for list_path in full_image_list_path:
                frame_ids_per_chunk.append(avtd.add_to_db(temp_db, current_metadata, frame_list_path=list_path))

            for k, (list_path, full_db, chunk_output_model, frame_ids) in enumerate(zip(full_image_list_path,
                                                                                        full_dbs,
                                                                                        chunk_output_models,
                                                                                        frame_ids_per_chunk)):
                print("Localizing Chunk {}/{}".format(k + 1, len(full_dbs)))
                chunk_output_model.makedirs_p()
                lowfps_db.copy(full_db)
                colmap.db = full_db
                avtd.add_to_db(full_db, current_metadata, frame_list_path=list_path, input_frame_ids=frame_ids)
                colmap.extract_features(image_list=list_path, fine=args.fine_sift_features)
                colmap.match(method="sequential", vocab_tree=env["indexed_vocab_tree"])
                colmap.register_images(output=chunk_output_model, input=video_output_model)
                colmap.adjust_bundle(output=chunk_output_model, input=chunk_output_model)
            for chunk in chunk_output_models:
                colmap.merge_models(output=video_output_model, input1=video_output_model, input2=chunk)
            final_output_model.makedirs_p()
            empty = not evfm.extract_video(input=video_output_model,
                                           output=final_output_model,
                                           video_metadata_path=current_metadata,
                                           output_format=".bin" if args.triangulate else ".txt")

            if empty:
                print("Error, empty localization, will try map from video")
                continue
                # colmap.db = lowfps_db
                # colmap.map(output_model=video_output_model, start_frame_id=added_frames[int(len(added_frames)/2)])
                # colmap.align_model(output_model=video_output_model,
                #                    input_model=video_output_model / "0",
                #                    ref_images=current_video_folder / "georef.txt")
                # colmap.db = full_db
                # colmap.register_images(output_model=video_output_model, input_model=video_output_model)
                # colmap.adjust_bundle(output_model=video_output_model, input_model=video_output_model)
                # empty = not evfm.extract_video(input_model=video_output_model,
                #                                output_model=final_output_model,
                #                                video_metadata_path=current_metadata,
                #                                output_format=".txt")
                # if empty:
                #     print("Error could not map anything, aborting this video")
                #     continue

            if args.triangulate:
                i_pv += 1
                print_step(i, "Re-Alignment of triangulated points with Lidar point cloud")

                colmap.triangulate_points(final_output_model, final_output_model)
                colmap.export_model(final_output_model, final_output_model, output_type="TXT")
                ply_name = final_output_model / "georef_{}.ply".format(v.namebase)
                matrix_name = final_output_model / "georef_maxtrix_{}.txt".format(v.namebase)
                colmap.export_model(ply_name, final_output_model, output_type="PLY")
                pcl_util.register_reconstruction(georef=ply_name, lidar=env["lidar_ply"],
                                                 output_matrix=matrix_name, output_cloud=env["lidar_ply"],
                                                 max_distance=10)
                matrix = np.fromfile(matrix_name, sep=" ").reshape(4, 4)

            output_images_folder = args.output_folder / "Images" / v.namebase
            output_images_folder.makedirs_p()
            current_video_folder.merge_tree(output_images_folder)

            if args.save_space:
                for file in current_video_folder.files():
                    if file not in existing_images:
                        file.remove()

    for j, v in enumerate(env["videos_list"]):
        output_images_folder = args.output_folder / "Images" / v.namebase
        current_video_folder = env["videos_output_folders"][v]
        current_metadata = current_video_folder / "metadata.csv"
        video_output_model = env["video_recon"] / v.namebase
        final_output_model = video_output_model / "final"
        model_length = len(read_images_text(final_output_model / "images.txt"))
        if model_length < 2:
            continue

        final_lidar = final_output_model / "aligned_lidar.mlp"
        final_occlusions = final_output_model / "occlusions.mlp"
        final_splats = final_output_model / "splats.mlp"
        specific_matrix_path = final_output_model / "matrix.txt"
        if specific_matrix_path.isfile():
            current_matrix = np.linalg.inv(np.fromfile(specific_matrix_path, sep=" ").reshape(4, 4))
        else:
            current_matrix = matrix

        mxw.apply_transform_to_project(env["lidar_mlp"], final_lidar, current_matrix)
        mxw.create_project(final_occlusions, [env["occlusion_ply"]], transforms=[current_matrix])
        mxw.create_project(final_splats, [env["splats_ply"]], transforms=[current_matrix])
        output_vizualisation_folder = args.output_folder / "video" / v.namebase

        if args.resume_work and output_vizualisation_folder.isdir():
            continue

        print("Creating GT on video {} [{}/{}]".format(v, j+1, len(env["videos_list"])))
        i_pv = 0

        i_pv += 1
        print_step(i, "Creating Ground truth data with ETH3D")

        # eth3d.create_ground_truth(final_lidar, final_occlusions,
        #                           final_splats, final_output_model,
        #                           args.output_folder)
        eth3d.create_ground_truth(env["aligned_mlp"], env["occlusion_mlp"],
                                  env["splats_mlp"], final_output_model,
                                  args.output_folder)
        output_vizualisation_folder.makedirs_p()

        i_pv += 1
        print_step(i_pv, "Create video with GT vizualisation")

        fps = pd.read_csv(current_metadata)["framerate"].iloc[0]
        vd.process_viz(args.output_folder / "ground_truth_depth" / v.namebase,
                       output_images_folder,
                       args.output_folder / "occlusion_depth" / v.namebase,
                       output_vizualisation_folder,
                       video=True, fps=fps, downscale=4, threads=8, **env)
        i_pv += 1
        print_step(i_pv, "Convert to KITTI format")


if __name__ == '__main__':
    main()
