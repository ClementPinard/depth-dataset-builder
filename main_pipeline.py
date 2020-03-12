from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from global_options import add_global_options
import meshlab_xml_writer as mxw
import add_video_to_db as avtd
import extract_video_from_model as evfm
from path import Path
import numpy as np
import videos_to_colmap as v2c
import generate_sky_masks as gsm
import las2ply
import rawpy
import imageio

global_steps = ["Point Cloud Preparation",
                "Pictures preparation",
                "Extracting Videos and selecting optimal frames for a thorough scan",
                "First thorough photogrammetry",
                "Alignment of photogrammetric reconstruction with Lidar point cloud",
                "Occlusion Mesh computing"]

pre_vid_steps = ["Full video extraction"
                 "Sky mask generation",
                 "Complete photogrammetry with video at 1 fps",
                 "Localizing remaining frames",
                 "Creating Ground truth data",
                 "Create video with GT vizualisation"]

parser = ArgumentParser(description='Main pipeline, from LIDAR pictures and videos to GT depth enabled videos',
                        formatter_class=ArgumentDefaultsHelpFormatter)

main_parser = parser.add_argument_group("Main options")
main_parser.add_argument('--input_folder', metavar='PATH', default=Path("."), type=Path,
                         help="Folder with LAS point cloud, videos, and images")
main_parser.add_argument('--workspace', metavar='PATH', default=Path("."),
                         help='path to workspace where COLMAP operations will be done', type=Path)
main_parser.add_argument('--output_folder', metavar='PATH', default=Path("."),
                         help='path to output folder : must be big !')
main_parser.add_argument('--skip_step', metavar="N", nargs="*", default=[], type=int)
main_parser.add_argument('--begin_step', metavar="N", type=int, default=None)
main_parser.add_argument('--show_steps', action="store_true")
main_parser.add_argument('-v', '--verbose', action="count", default=0)
main_parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"])
main_parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"])
main_parser.add_argument('--raw_ext', nargs='+', default=[".ARW", ".NEF", ".DNG"])
main_parser.add_argument('--save_space', action="store_true")

ve_parser = parser.add_argument_group("Video extractor")
ve_parser.add_argument('--total_frames', default=500, type=int)
ve_parser.add_argument('--orientation_weight', default=1, type=float)
ve_parser.add_argument('--resolution_weight', default=1, type=float)
ve_parser.add_argument('--num_neighbours', default=10, type=int)
ve_parser.add_argument('--system', default="epsg:2154")

exec_parser = parser.add_argument_group("Executable files")
exec_parser.add_argument('--nw', default="native-wrapper.sh",
                         help="native-wrapper.sh file location")
exec_parser.add_argument("--colmap", default="colmap",
                         help="colmap exec file location")
exec_parser.add_argument("--eth3d", default="../dataset-pipeline/build",
                         type=Path, help="ETH3D detaset pipeline exec files folder location")
exec_parser.add_argument("--ffmpeg", default="ffmpeg")
exec_parser.add_argument("--pcl_util", default="pcl_util/build", type=Path)

video_registration_parser = parser.add_argument_group("Video Registration")
video_registration_parser.add_argument("--vocab_tree", type=Path, default="vocab_tree_flickr100K_words256K.bin")


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


def convert_point_cloud(pointclouds, lidar_path, verbose, eth3d, pcl_util, **env):
    converted_clouds = []
    centroids = []
    for pc in pointclouds:
        ply, centroid = las2ply.load_and_convert(input_file=pc,
                                                 output_folder=lidar_path,
                                                 verbose=verbose >= 1)
        centroids.append(centroid)
        eth3d.clean_pointcloud(ply, filter=(6, 2))
        pcl_util.subsample(input_file=ply + ".inliers.ply", output_file=ply + "_subsampled.ply", resolution=0.1)

        converted_clouds.append(ply + "_subsampled.ply")
    return converted_clouds, centroids[0]


def extract_pictures_to_workspace(input_folder, image_path, workspace, colmap, raw_ext, pic_ext, **env):
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
    colmap.extract_features(per_sub_folder=True, fine=False)
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

    env["aligned_mlp"] = env["workspace"] / "aligned_model.mlp"
    env["occlusion_ply"] = env["lidar_path"] / "occlusion_model.ply"
    env["splats_ply"] = env["lidar_path"] / "splats_model.ply"
    env["occlusion_mlp"] = env["lidar_path"] / "occlusions.mlp"
    env["georefrecon_ply"] = env["georef_recon"] / "georef_reconstruction.ply"


def main():
    args = parser.parse_args()
    env = vars(args)
    if args.show_steps:
        print_workflow()
    if args.begin_step is not None:
        args.skip_step += list(range(args.begin_step))
    check_input_folder(args.input_folder)
    prepare_workspace(args.workspace, env)
    colmap = Colmap(db=args.workspace/"thorough_scan.db",
                    image_path=env["image_path"],
                    mask_path=env["mask_path"],
                    binary=args.colmap, quiet=args.verbose < 1)
    env["colmap"] = colmap
    ffmpeg = FFMpeg(args.ffmpeg, quiet=args.verbose < 2)
    env["ffmpeg"] = ffmpeg
    pdraw = PDraw(args.nw, quiet=args.verbose < 2)
    env["pdraw"] = pdraw
    eth3d = ETH3D(args.eth3d, env["image_path"], quiet=args.verbose < 1)
    env["eth3d"] = eth3d
    pcl_util = PCLUtil(args.pcl_util)
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
        path_lists, env["videos_output_folders"] = extract_videos_to_workspace(**env)
        if path_lists is not None:
            with open(env["video_frame_list_thorough"], "w") as f:
                f.write("\n".join(path_lists["thorough"]))
            with open(env["georef_frames_list"], "w") as f:
                f.write("\n".join(path_lists["georef"]))
            for v in env["videos_list"]:
                with open(env["videos_output_folders"][v] / "to_scan.txt", "w") as f:
                    f.write("\n".join(path_lists[v]))
    else:
        by_basename = {v.basename(): v for v in env["videos_list"]}
        for folder in env["video_path"].walkdirs():
            video_name = folder.basename()
            if video_name in by_basename.keys():
                env["videos_output_folders"][by_basename[video_name]] = folder

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)
        gsm.process_folder(folder_to_process=env["video_path"], **env)
        colmap.extract_features(image_list=env["video_frame_list_thorough"], fine=False)
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

        mxw.create_project(env["workspace"] / "unaligned.mlp", [env["georefrecon_ply"]] + env["pointclouds"])
        eth3d.align_with_ICP(env["workspace"] / "unaligned.mlp", env["aligned_mlp"], scales=5)
        mxw.remove_mesh_from_project(env["aligned_mlp"], env["aligned_mlp"], 0)

    i, s = next(i_global_steps)
    if i + 1 not in args.skip_step:
        print_step(i, s)

        with_normals_path = env["lidar_path"] / "with_normals.ply"
        eth3d.compute_normals(with_normals_path, env["aligned_mlp"], neighbor_radius=0.2)
        pcl_util.triangulate_mesh(env["occlusion_ply"], with_normals_path, resolution=0.2)
        eth3d.create_splats(env["splats_ply"], with_normals_path, env["occlusion_ply"], threshold=0.1)
        mxw.create_project(env["occlusion_mlp"], [env["occlusion_ply"], env["splats_ply"]])
        if args.save_space:
            with_normals_path.remove()

    for v in env["videos_list"]:
        i_pv_steps = enumerate(pre_vid_steps)
        print("Now working on video {}".format(v))
        current_video_folder = env["videos_output_folders"][v]
        former_db = colmap.db
        current_db = current_video_folder / "video.db"
        current_metadata = current_video_folder / "metadata.csv"
        former_db.copy(current_db)
        colmap.db = current_db

        i, s = next(i_pv_steps)
        print_step(i, s)
        existing_images = list(current_video_folder.files())
        ffmpeg.extract_images(v, current_video_folder)

        i, s = next(i_pv_steps)
        print_step(i, s)
        gsm.process_folder(folder_to_process=current_video_folder, **env)

        i, s = next(i_pv_steps)
        print_step(i, s)
        image_list_path = current_video_folder / "to_scan.txt"
        avtd.add_to_db(current_db, current_metadata, image_list_path)
        colmap.extract_features(image_list=image_list_path, fine=False)

        i, s = next(i_pv_steps)
        print_step(i, s)

        colmap.match(method="sequential", vocab_tree=args.vocab_tree)
        video_output_model = env["video_recon"] / v.basename()
        video_output_model.makedirs_p()
        colmap.map(output_model=video_output_model, input_model=env["georef"])
        colmap.align_model(output_model=video_output_model,
                           input_model=video_output_model,
                           ref_images=env["georef_frames_list"])

        i, s = next(i_pv_steps)
        print_step(i, s)
        avtd.add_to_db(current_db, current_metadata, frame_list_path=None)
        colmap.extract_features(image_list=current_video_folder / "to_scan.txt", fine=False)
        colmap.match(method="sequential", vocab_tree=args.vocab_tree)
        colmap.register_images(output_model=video_output_model, input_model=video_output_model)
        colmap.adjust_bundle(output_model=video_output_model, input_model=video_output_model)
        colmap.align_model(output_model=video_output_model,
                           input_model=video_output_model,
                           ref_images=env["georef_frames_list"])
        final_output_model = video_output_model / "final"
        final_output_model.makedirs_p()
        evfm.extract_video(input_model=video_output_model,
                           output_model=final_output_model,
                           video_metadata_path=current_metadata)

        i, s = next(i_pv_steps)
        print_step(i, s)
        eth3d.create_ground_truth(env["aligned_mlp"], env["occlusion_mlp"], final_output_model, args.output_folder)

        if args.save_space:
            for file in current_video_folder.files():
                if file not in existing_images:
                    file.remove()


if __name__ == '__main__':
    main()
