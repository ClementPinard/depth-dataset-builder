import las2ply
import rawpy
import imageio

import numpy as np
import pandas as pd
from pyproj import Proj
from edit_exif import get_gps_location

from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from cli_utils import set_argparser, print_step, print_workflow
from video_localization import localize_video, generate_GT
import meshlab_xml_writer as mxw
import videos_to_colmap as v2c
import generate_sky_masks as gsm


def prepare_point_clouds(pointclouds, lidar_path, verbose, eth3d, pcl_util, SOR, pointcloud_resolution, **env):
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
        # pcl_util.subsample(input_file=ply.stripext() + "_filtered.ply",
        #                    output_file=ply.stripext() + "_subsampled.ply",
        #                    resolution=pointcloud_resolution)

        # converted_clouds.append(ply.stripext() + "_subsampled.ply")
        converted_clouds.append(ply.stripext() + "_filtered.ply")
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


def extract_pictures_to_workspace(input_folder, image_path, workspace, colmap,
                                  raw_ext, pic_ext, more_sift_features, **env):
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
    colmap.extract_features(per_sub_folder=True, more=more_sift_features)
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
                             "Pictures/Videos", "Thorough/0", "Thorough/georef", "Thorough/georef_full",
                             "Videos_reconstructions"],
                            ["lidar_path", "image_path", "mask_path",
                             "video_path", "thorough_recon", "georef_recon", "georef_full_recon",
                             "video_recon"]):
        (path / dirname).makedirs_p()
        env[key] = path / dirname

    env["thorough_db"] = path / "scan_thorough.db"
    env["video_frame_list_thorough"] = env["image_path"] / "video_frames_for_thorough_scan.txt"
    env["georef_frames_list"] = env["image_path"] / "georef.txt"

    env["lidar_mlp"] = env["workspace"] / "lidar.mlp"
    env["with_normals_path"] = env["lidar_path"] / "with_normals.ply"
    env["aligned_mlp"] = env["workspace"] / "aligned_model.mlp"
    env["occlusion_ply"] = env["lidar_path"] / "occlusion_model.ply"
    env["splats_ply"] = env["lidar_path"] / "splats_model.ply"
    env["occlusion_mlp"] = env["lidar_path"] / "occlusions.mlp"
    env["splats_mlp"] = env["lidar_path"] / "splats.mlp"
    env["georefrecon_ply"] = env["georef_recon"] / "georef_reconstruction.ply"
    env["matrix_path"] = env["workspace"] / "matrix_thorough.txt"
    env["indexed_vocab_tree"] = env["workspace"] / "vocab_tree_thorough.bin"


def prepare_video_workspace(video_name, video_frames_folder, output_folder, video_recon, image_path, **env):
    video_env = {video_name: video_name,
                 video_frames_folder: video_frames_folder}
    relative_path_folder = video_frames_folder.relpath(image_path)
    video_env["lowfps_db"] = video_frames_folder / "video_low_fps.db"
    video_env["metadata"] = video_frames_folder / "metadata.csv"
    video_env["lowfps_image_list_path"] = video_frames_folder / "lowfps.txt"
    video_env["chunk_image_list_paths"] = video_frames_folder.files("full_chunk_*.txt")
    video_env["chunk_dbs"] = [video_frames_folder / fp.namebase + ".db" for fp in video_env["chunk_image_list_paths"]]
    colmap_root = video_recon / relative_path_folder
    video_env["colmap_models_root"] = colmap_root
    video_env["full_model"] = colmap_root
    video_env["lowfps_model"] = colmap_root / "lowfps"
    num_chunks = len(video_env["chunk_image_list_paths"])
    video_env["chunk_models"] = [colmap_root / "chunk_{}".format(index) for index in range(num_chunks)]
    video_env["final_model"] = colmap_root / "final"
    output = {}
    output["images_root_folder"] = output_folder / "images"
    output["video_frames_folder"] = output["images_root_folder"] / relative_path_folder
    output["viz_folder"] = output_folder / "video" / relative_path_folder
    output["model_folder"] = output_folder / "models" / relative_path_folder
    output["final_model"] = output["model_folder"] / "final"
    output["video_fps"] = pd.read_csv(video_env["metadata"])["framerate"].values[0]
    video_env["output_env"] = output
    video_env["already_localized"] = env["resume_work"] and output["images_folder"].isdir()
    video_env["GT_already_done"] = env["resume_work"] and (output_folder / "groundtruth_depth" / video_name.namebase).isdir()
    return video_env


def main():
    args = set_argparser().parse_args()
    env = vars(args)
    if args.show_steps:
        print_workflow()
    if args.add_new_videos:
        args.skip_step += [1, 2, 4, 5, 6]
    if args.begin_step is not None:
        args.skip_step += list(range(args.begin_step))
    check_input_folder(args.input_folder)
    args.workspace = args.workspace.abspath()
    prepare_workspace(args.workspace, env)
    colmap = Colmap(db=env["thorough_db"],
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
    eth3d = ETH3D(args.eth3d, args.output_folder / "Images", quiet=args.verbose < 1, logfile=args.log)
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
        print_step(i, "Occlusion Mesh computing")
        eth3d.compute_normals(env["with_normals_path"], env["lidar_mlp"], neighbor_radius=args.normal_radius)
        pcl_util.triangulate_mesh(env["occlusion_ply"], env["with_normals_path"], resolution=args.mesh_resolution)
        eth3d.create_splats(env["splats_ply"], env["with_normals_path"], env["occlusion_ply"], threshold=args.splat_threshold)

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
        path_lists, env["videos_frames_folders"] = extract_videos_to_workspace(fps=args.lowfps, **env)
        if path_lists is not None:
            with open(env["video_frame_list_thorough"], "w") as f:
                f.write("\n".join(path_lists["thorough"]["frames"]))
            with open(env["georef_frames_list"], "w") as f:
                f.write("\n".join(existing_georef) + "\n")
                f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
            for v in env["videos_list"]:
                video_folder = env["videos_frames_folders"][v]
                with open(video_folder / "lowfps.txt", "w") as f:
                    f.write("\n".join(path_lists[v]["frames_lowfps"]) + "\n")
                with open(video_folder / "georef.txt", "w") as f:
                    f.write("\n".join(existing_georef) + "\n")
                    f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
                    f.write("\n".join(path_lists[v]["georef_lowfps"]) + "\n")
                for j, l in enumerate(path_lists[v]["frames_full"]):
                    with open(video_folder / "full_chunk_{}.txt".format(j), "w") as f:
                        f.write("\n".join(l) + "\n")
    else:
        env["videos_frames_folders"] = {}
        by_name = {v.namebase: v for v in env["videos_list"]}
        for folder in env["video_path"].walkdirs():
            video_name = folder.basename()
            if video_name in by_name.keys():
                env["videos_frames_folders"][by_name[video_name]] = folder
    env["videos_workspaces"] = {}
    for v, frames_folder in env["videos_frames_folders"].items():
        env["videos_workspaces"][v] = prepare_video_workspace(v, frames_folder, **env)

    i += 1
    if i not in args.skip_step:
        print_step(i, "First thorough photogrammetry")
        gsm.process_folder(folder_to_process=env["video_path"], **env)
        colmap.extract_features(image_list=env["video_frame_list_thorough"], more=args.more_sift_features)
        colmap.index_images(vocab_tree_output=env["indexed_vocab_tree"], vocab_tree_input=args.vocab_tree)
        colmap.match()
        colmap.map(output=env["thorough_recon"].parent)

    i += 1
    if i not in args.skip_step:
        print_step(i, "Alignment of photogrammetric reconstruction with GPS")

        colmap.align_model(output=env["georef_recon"],
                           input=env["thorough_recon"],
                           ref_images=env["georef_frames_list"])
        env["georef_recon"].merge_tree(env["georef_full_recon"])
        if args.dense:
            print_step("{} (bis)".format(i), "Point cloud densificitation")
            dense_workspace = env["thorough_recon"].parent/"dense"
            colmap.undistort(input=env["georef_recon"], output=dense_workspace)
            colmap.dense_stereo(workspace=dense_workspace)
            colmap.stereo_fusion(workspace=dense_workspace, output=env["georefrecon_ply"])
        else:
            colmap.export_model(output=env["georefrecon_ply"],
                                input=env["georef_recon"])
    if args.inspect_dataset:
        georef_mlp = env["georef_recon"]/"georef_recon.mlp"
        mxw.create_project(georef_mlp, [env["georefrecon_ply"]])
        colmap.export_model(output=env["georef_recon"],
                            input=env["georef_recon"],
                            output_type="TXT")
        eth3d.inspect_dataset(scan_meshlab=georef_mlp,
                              colmap_model=env["georef_recon"],
                              image_path=env["image_path"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Registration of photogrammetric reconstruction with respect to Lidar Point Cloud")
        if args.registration_method == "simple":
            pcl_util.register_reconstruction(georef=env["georefrecon_ply"],
                                             lidar=env["with_normals_path"],
                                             output_matrix=env["matrix_path"],
                                             max_distance=10)
        elif args.registration_method == "eth3d":
            temp_mlp = env["lidar_mlp"].stripext() + "_registered.mlp"
            mxw.add_mesh_to_project(env["lidar_mlp"], temp_mlp, env["georefrecon_ply"], index=0)
            eth3d.align_with_ICP(temp_mlp, temp_mlp, scales=5)
            mxw.remove_mesh_from_project(temp_mlp, temp_mlp, 0)
            print(mxw.get_mesh(temp_mlp, index=0)[0])
            matrix = np.linalg.inv(mxw.get_mesh(temp_mlp, index=0)[0])
            np.savetxt(env["matrix_path"], matrix)

        elif args.registration_method == "interactive":
            input("Get transformation matrix and paste it in the file {}. When done, press ENTER".format(env["matrix_path"]))
    if env["matrix_path"].isfile():
        env["global_registration_matrix"] = np.linalg.inv(np.fromfile(env["matrix_path"], sep=" ").reshape(4, 4))
    else:
        print("Error, no registration matrix can be found")
        env["global_registration_matrix"] = np.eye(4)

    mxw.apply_transform_to_project(env["lidar_mlp"], env["aligned_mlp"], env["global_registration_matrix"])
    mxw.create_project(env["occlusion_mlp"], [env["occlusion_ply"]], transforms=[env["global_registration_matrix"]])
    mxw.create_project(env["splats_mlp"], [env["splats_ply"]], transforms=[env["global_registration_matrix"]])

    if args.inspect_dataset:
        eth3d.inspect_dataset(scan_meshlab=env["aligned_mlp"],
                              colmap_model=env["georef_recon"],
                              image_path=env["image_path"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Video localization")
        for j, v in enumerate(env["videos_list"]):
            print("\n\nNow working on video {} [{}/{}]".format(v, j + 1, len(env["videos_list"])))
            video_env = env["videos_workspaces"][v]
            localize_video(video_name=v,
                           video_frames_folder=env["videos_frames_folders"][v],
                           video_index=j+1,
                           num_videos=len(env["videos_list"]),
                           **video_env, **env)

    i += 1
    if i not in args.skip_step:
        print_step(i, "Groud Truth generation")
        for j, v in enumerate(env["videos_list"]):
            video_env = env["videos_workspaces"][v]

            generate_GT(video_name=v, GT_already_done=video_env["GT_already_done"],
                        video_index=j+1,
                        num_videos=len(env["videos_list"]),
                        **video_env["output_env"], **env)


if __name__ == '__main__':
    main()
