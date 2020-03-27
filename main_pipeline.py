from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from global_options import add_global_options
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

global_steps = ["Point Cloud Preparation",
                "Pictures preparation",
                "Extracting Videos and selecting optimal frames for a thorough scan",
                "First thorough photogrammetry",
                "Alignment of photogrammetric reconstruction with Lidar point cloud",
                "Occlusion Mesh computing"]

pre_vid_steps = ["Full video extraction",
                 "Sky mask generation",
                 "Complete photogrammetry with video at 1 fps",
                 "Localizing remaining frames",
                 "Creating Ground truth data",
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
main_parser.add_argument('--fine_sift_features', action="store_true")
main_parser.add_argument('--save_space', action="store_true")
main_parser.add_argument('--add_new_videos', action="store_true")

pcp_parser = parser.add_argument_group("PointCLoud preparation")
pcp_parser.add_argument("--pointcloud_resolution", default=0.1, type=float)
pcp_parser.add_argument("--SOR", default=[6, 2], nargs=2, type=int)

ve_parser = parser.add_argument_group("Video extractor")
ve_parser.add_argument('--total_frames', default=500, type=int)
ve_parser.add_argument('--orientation_weight', default=1, type=float)
ve_parser.add_argument('--resolution_weight', default=1, type=float)
ve_parser.add_argument('--num_neighbours', default=10, type=int)
ve_parser.add_argument('--system', default="epsg:2154")

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
    for i, s in enumerate(pre_vid_steps):
        print("\t{}:\t{}".format(i, s))


def print_step(step_number, step_name):
        print("=================")
        print("Step {}".format(step_number + 1))
        print(step_name)
        print("=================")


def convert_point_cloud(pointclouds, lidar_path, verbose, eth3d, pcl_util, pointcloud_resolution, save_space, **env):
    converted_clouds = []
    centroids = []
    for pc in pointclouds:
        ply, centroid = las2ply.load_and_convert(input_file=pc,
                                                 output_folder=lidar_path,
                                                 verbose=verbose >= 1)
        centroids.append(centroid)
        eth3d.clean_pointcloud(ply, filter=(6, 2))
        pcl_util.subsample(input_file=ply + ".inliers.ply", output_file=ply.stripext() + "_subsampled.ply", resolution=pointcloud_resolution)
        if save_space:
            (ply + ".inliers.ply").remove()
            (ply + ".outliers.ply").remove()
            ply.remove()

        converted_clouds.append(ply.stripext() + "_subsampled.ply")
    temp_mlp = env["workspace"] / "lidar_unaligned.mlp"
    mxw.create_project(temp_mlp, converted_clouds, labels=None, transforms=None)
    if len(converted_clouds) > 1:
        eth3d.align_with_ICP(temp_mlp, env["lidar_mlp"], scales=5)
    else:
        temp_mlp.move(env["lidar_mlp"])

    return converted_clouds, centroids[0]


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
    i_global_steps = enumerate(global_steps)

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)
        env["pointclouds"], env["centroid"] = convert_point_cloud(input_pointclouds, **env)
    else:
        env["pointclouds"] = env["lidar_path"].files("*inliers.ply")
        centroid_path = sorted(env["lidar_path"].files("*_centroid.txt"))[0]
        env["centroid"] = np.loadtxt(centroid_path)

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)
        env["existing_pictures"] = extract_pictures_to_workspace(**env)
    else:
        env["existing_pictures"] = sum((list(env["image_path"].walkfiles('*{}'.format(ext))) for ext in env["pic_ext"]), [])

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)
        existing_georef = extract_gps_and_path(**env)
        path_lists, env["videos_output_folders"] = extract_videos_to_workspace(**env)
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
                with open(video_folder / "full.txt", "w") as f:
                    f.write("\n".join(path_lists[v]["frames_full"]) + "\n")
    else:
        env["videos_output_folders"] = {}
        by_name = {v.namebase: v for v in env["videos_list"]}
        for folder in env["video_path"].walkdirs():
            video_name = folder.basename()
            if video_name in by_name.keys():
                env["videos_output_folders"][by_name[video_name]] = folder

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)
        gsm.process_folder(folder_to_process=env["video_path"], **env)
        colmap.extract_features(image_list=env["video_frame_list_thorough"], fine=args.fine_sift_features)
        colmap.match()
        colmap.map(output_model=env["thorough_recon"].parent)

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)

        colmap.align_model(output_model=env["georef_recon"],
                           input_model=env["thorough_recon"],
                           ref_images=env["georef_frames_list"])

        colmap.export_model(output_ply=env["georefrecon_ply"],
                            input_model=env["georef_recon"])

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)

        with_normals_path = env["lidar_path"] / "with_normals.ply"
        eth3d.compute_normals(with_normals_path, env["lidar_mlp"], neighbor_radius=args.normal_radius)
        pcl_util.triangulate_mesh(env["occlusion_ply"], with_normals_path, resolution=args.mesh_resolution)
        eth3d.create_splats(env["splats_ply"], with_normals_path, env["occlusion_ply"], threshold=args.splat_threshold)

        matrix_path = env["workspace"] / "matrix_thorough.txt"
        pcl_util.register_reconstruction(georef=env["georefrecon_ply"],
                                         lidar=with_normals_path,
                                         output_matrix=matrix_path,
                                         output_cloud=env["lidar_ply"],
                                         max_distance=10)
        matrix = np.fromfile(matrix_path, sep=" ").reshape(4, 4)
        mxw.apply_transform_to_project(env["lidar_mlp"], env["aligned_mlp"], matrix)
        mxw.create_project(env["occlusion_mlp"], [env["occlusion_ply"]], transforms=[matrix])
        mxw.create_project(env["splats_mlp"], [env["splats_ply"]], transforms=[matrix])

    for v in env["videos_list"]:
        i_pv_steps = enumerate(pre_vid_steps)
        print("Now working on video {}".format(v))
        current_video_folder = env["videos_output_folders"][v]
        thorough_db = args.workspace / "thorough_scan.db"
        lowfps_db = current_video_folder / "video1fps.db"
        full_db = current_video_folder / "video_full.db"
        current_metadata = current_video_folder / "metadata.csv"
        thorough_db.copy(lowfps_db)
        map_image_list_path = current_video_folder / "to_scan.txt"
        full_image_list_path = current_video_folder / "full.txt"
        colmap.db = lowfps_db

        i, s = next(i_pv_steps)
        print_step(i, s)
        if args.save_space:
            existing_images = list(current_video_folder.files())
            ffmpeg.extract_images(v, current_video_folder)
        else:
            print("Already Done.")

        i, s = next(i_pv_steps)
        print_step(i, s)
        gsm.process_folder(folder_to_process=current_video_folder, **env)

        i, s = next(i_pv_steps)
        print_step(i, s)
        added_frames = avtd.add_to_db(lowfps_db, current_metadata, map_image_list_path)
        colmap.extract_features(image_list=map_image_list_path, fine=args.fine_sift_features)
        colmap.match(method="sequential", vocab_tree=args.vocab_tree)
        video_output_model = env["video_recon"] / v.namebase
        video_output_model.makedirs_p()
        colmap.map(output_model=video_output_model, input_model=env["georef_recon"])
        # when colmap map is called, the model is normalized so we have georegister it again
        colmap.align_model(output_model=video_output_model,
                           input_model=video_output_model,
                           ref_images=env["georef_frames_list"])

        i, s = next(i_pv_steps)
        print_step(i, s)
        lowfps_db.copy(full_db)
        colmap.db = full_db
        avtd.add_to_db(full_db, current_metadata, frame_list_path=None)
        colmap.extract_features(image_list=full_image_list_path, fine=args.fine_sift_features)
        colmap.match(method="sequential", vocab_tree=args.vocab_tree)
        colmap.register_images(output_model=video_output_model, input_model=video_output_model)
        colmap.adjust_bundle(output_model=video_output_model, input_model=video_output_model)
        final_output_model = video_output_model / "final"
        final_output_model.makedirs_p()
        empty = not evfm.extract_video(input_model=video_output_model,
                                       output_model=final_output_model,
                                       video_metadata_path=current_metadata,
                                       output_format=".bin")

        if empty:
            print("Error, empty localization, will try map from video")
            colmap.db = lowfps_db
            colmap.map(output_model=video_output_model, start_frame_id=added_frames[0])
            colmap.align_model(output_model=video_output_model,
                               input_model=video_output_model / "0",
                               ref_images=current_video_folder / "georef.txt")
            colmap.db = full_db
            colmap.register_images(output_model=video_output_model, input_model=video_output_model)
            colmap.adjust_bundle(output_model=video_output_model, input_model=video_output_model)
            empty = not evfm.extract_video(input_model=video_output_model,
                                           output_model=final_output_model,
                                           video_metadata_path=current_metadata,
                                           output_format=".bin")
        i, s = next(i_pv_steps)
        print_step(i, s)

        colmap.triangulate_points(final_output_model, final_output_model)
        colmap.export_model(final_output_model, final_output_model, output_type="TXT")
        ply_name = final_output_model / "georef_{}.ply".format(v.namebase)
        matrix_name = final_output_model / "georef_maxtrix_{}.txt".format(v.namebase)
        colmap.export_model(ply_name, final_output_model, output_type="PLY")
        pcl_util.register_reconstruction(georef=ply_name,
                                         lidar=env["lidar_ply"],
                                         output_matrix=matrix_name,
                                         output_cloud=env["lidar_ply"],
                                         max_distance=10)
        matrix = np.fromfile(matrix_name, sep=" ").reshape(4, 4)
        final_lidar = final_output_model / "aligned_lidar.mlp"
        final_occlusions = final_output_model / "occlusions.mlp"
        final_splats = final_output_model / "splats.mlp"
        mxw.apply_transform_to_project(env["aligned_mlp"], final_lidar, matrix)
        mxw.apply_transform_to_project(env["occlusion_mlp"], final_occlusions, matrix)
        mxw.apply_transform_to_project(env["splats_mlp"], final_splats, matrix)

        i, s = next(i_pv_steps)
        print_step(i, s)
        output_images_folder = args.output_folder / "Images" / v.namebase
        output_images_folder.makedirs_p()
        current_video_folder.merge_tree(output_images_folder)

        eth3d.create_ground_truth(final_lidar, final_occlusions,
                                  final_splats, final_output_model,
                                  args.output_folder)
        output_vizualisation_folder = args.output_folder / "video" / v.namebase
        output_vizualisation_folder.makedirs_p()

        fps = pd.read_csv(current_metadata)["framerate"].iloc[0]
        vd.process_viz(args.output_folder / "ground_truth_depth" / v.namebase,
                       output_images_folder,
                       args.output_folder / "occlusion_depth" / v.namebase,
                       output_vizualisation_folder,
                       video=True, fps=fps, downscale=4, threads=8, **env)

        if args.save_space:
            for file in current_video_folder.files():
                if file not in existing_images:
                    file.remove()
            colmap.db.remove()


if __name__ == '__main__':
    main()
