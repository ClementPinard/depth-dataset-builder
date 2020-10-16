import las2ply
import numpy as np
from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from cli_utils import set_argparser, print_step, print_workflow
from video_localization import localize_video, generate_GT
import meshlab_xml_writer as mxw
import prepare_images as pi
import prepare_workspace as pw


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


def main():
    args = set_argparser().parse_args()
    env = vars(args)
    if args.show_steps:
        print_workflow()
        return
    if args.add_new_videos:
        args.skip_step += [1, 2, 4, 5, 6]
    if args.begin_step is not None:
        args.skip_step += list(range(args.begin_step))
    pw.check_input_folder(args.input_folder)
    args.workspace = args.workspace.abspath()
    pw.prepare_workspace(args.workspace, env)
    colmap = Colmap(db=env["thorough_db"],
                    image_path=env["image_path"],
                    mask_path=env["mask_path"],
                    dense_workspace=env["dense_workspace"],
                    binary=args.colmap,
                    verbose=args.verbose,
                    logfile=args.log)
    env["colmap"] = colmap
    ffmpeg = FFMpeg(args.ffmpeg, verbose=args.verbose, logfile=args.log)
    env["ffmpeg"] = ffmpeg
    pdraw = PDraw(args.nw, verbose=args.verbose, logfile=args.log)
    env["pdraw"] = pdraw
    eth3d = ETH3D(args.eth3d, args.raw_output_folder / "Images", args.max_occlusion_depth,
                  verbose=args.verbose, logfile=args.log)
    env["eth3d"] = eth3d
    pcl_util = PCLUtil(args.pcl_util, verbose=args.verbose, logfile=args.log)
    env["pcl_util"] = pcl_util

    las_files = (args.input_folder/"Lidar").files("*.las")
    ply_files = (args.input_folder/"Lidar").files("*.ply")
    input_pointclouds = las_files + ply_files
    env["videos_list"] = sum((list((args.input_folder/"Videos").walkfiles('*{}'.format(ext))) for ext in args.vid_ext), [])
    no_gt_folder = args.input_folder/"Videos"/"no_groundtruth"
    if no_gt_folder.isdir():
        env["videos_to_localize"] = [v for v in env["videos_list"] if not str(v).startswith(no_gt_folder)]
    else:
        env["videos_to_localize"] = env["videos_list"]

    i = 1
    if i not in args.skip_step:
        print_step(i, "Point Cloud Preparation")
        env["pointclouds"], env["centroid"] = prepare_point_clouds(input_pointclouds, **env)
        if env["centroid"] is not None:
            np.savetxt(env["centroid_path"], env["centroid"])
    else:
        if env["centroid_path"].isfile():
            env["centroid"] = np.loadtxt(env["centroid_path"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Pictures preparation")
        env["existing_pictures"] = pi.extract_pictures_to_workspace(**env)
    else:
        env["existing_pictures"] = sum((list(env["image_path"].walkfiles('*{}'.format(ext))) for ext in env["pic_ext"]), [])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Extracting Videos and selecting optimal frames for a thorough scan")
        existing_georef, env["centroid"] = pi.extract_gps_and_path(**env)
        env["videos_frames_folders"] = pi.extract_videos_to_workspace(existing_georef=existing_georef,
                                                                      fps=args.lowfps, **env)
    else:
        env["videos_frames_folders"] = {}
        by_name = {v.stem: v for v in env["videos_list"]}
        for folder in env["video_path"].walkdirs():
            video_name = folder.basename()
            if video_name in by_name.keys():
                env["videos_frames_folders"][by_name[video_name]] = folder
    env["videos_workspaces"] = {}
    for v, frames_folder in env["videos_frames_folders"].items():
        env["videos_workspaces"][v] = pw.prepare_video_workspace(v, frames_folder, **env)

    i += 1
    if i not in args.skip_step:
        print_step(i, "First thorough photogrammetry")
        env["thorough_recon"].makedirs_p()
        colmap.extract_features(image_list=env["video_frame_list_thorough"], more=args.more_sift_features)
        colmap.index_images(vocab_tree_output=env["indexed_vocab_tree"], vocab_tree_input=args.vocab_tree)
        colmap.match(method="vocab_tree", vocab_tree=env["indexed_vocab_tree"], max_num_matches=env["max_num_matches"])
        colmap.map(output=env["thorough_recon"], multiple_models=env["multiple_models"])
        thorough_model = pi.choose_biggest_model(env["thorough_recon"])
        colmap.adjust_bundle(thorough_model, thorough_model,
                             num_iter=100, refine_extra_params=True)
    else:
        thorough_model = pi.choose_biggest_model(env["thorough_recon"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Alignment of photogrammetric reconstruction with GPS")
        env["georef_recon"].makedirs_p()
        colmap.align_model(output=env["georef_recon"],
                           input=thorough_model,
                           ref_images=env["georef_frames_list"])
        if not (env["georef_frames_list"]/"images.bin").isfile():
            # GPS alignment failed, possibly because not enough GPS referenced images
            # Copy the original model without alignment
            (env["thorough_recon"] / "0").merge_tree(env["georef_recon"])
        env["georef_recon"].merge_tree(env["georef_full_recon"])
    if args.inspect_dataset:
        colmap.export_model(output=env["georef_recon"] / "georef_sparse.ply",
                            input=env["georef_recon"])
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
        print_step(i, "Video localization with respect to reconstruction")
        for j, v in enumerate(env["videos_to_localize"]):
            print("\n\nNow working on video {} [{}/{}]".format(v, j + 1, len(env["videos_to_localize"])))
            video_env = env["videos_workspaces"][v]
            localize_video(video_name=v,
                           video_frames_folder=env["videos_frames_folders"][v],
                           video_index=j+1,
                           step_index=i,
                           num_videos=len(env["videos_to_localize"]),
                           **video_env, **env)

    i += 1
    if i not in args.skip_step:
        print_step(i, "Full reconstruction point cloud densificitation")
        env["georef_full_recon"].makedirs_p()
        colmap.undistort(input=env["georef_full_recon"])
        colmap.dense_stereo()
        colmap.stereo_fusion(output=env["georefrecon_ply"])

    def get_matrix(path):
        if path.isfile():
            '''Note : We use the inverse matrix here, because in general, it's easier to register the reconstructed model into the lidar one,
            as the reconstructed will have less points, but in the end we need the matrix to apply to the lidar point to be aligned
            with the camera positions (ie the inverse)'''
            return np.linalg.inv(np.fromfile(env["matrix_path"], sep=" ").reshape(4, 4))
        else:
            print("Error, no registration matrix can be found, identity will be used")
            return np.eye(4)
    i += 1
    if i not in args.skip_step:
        print_step(i, "Registration of photogrammetric reconstruction with respect to Lidar Point Cloud")
        if args.registration_method == "eth3d":
            # Note : ETH3D doesn't register with scale, this might not be suitable for very large areas
            mxw.add_meshes_to_project(env["lidar_mlp"], env["aligned_mlp"], [env["georefrecon_ply"]], start_index=0)
            eth3d.align_with_ICP(env["aligned_mlp"], env["aligned_mlp"], scales=5)
            mxw.remove_mesh_from_project(env["aligned_mlp"], env["aligned_mlp"], 0)
            matrix = np.linalg.inv(mxw.get_mesh(env["aligned_mlp"], index=0)[0])
            np.savetxt(env["matrix_path"], matrix)

            ''' The new mlp is supposedly better than the one before because it was an ICP
            with N+1 models instead of just N so we replace it with the result on this scan
            by reversing the first transformation and getting back a mlp file with identity
            as first transform matrix'''
            mxw.apply_transform_to_project(env["aligned_mlp"], env["lidar_mlp"], matrix)
            env["global_registration_matrix"] = matrix
        else:
            if args.normals_method == "radius":
                eth3d.compute_normals(env["with_normals_path"], env["lidar_mlp"], neighbor_radius=args.normals_radius)
            else:
                eth3d.compute_normals(env["with_normals_path"], env["lidar_mlp"], neighbor_count=args.normals_neighbours)
            if args.registration_method == "simple":
                pcl_util.register_reconstruction(georef=env["georefrecon_ply"],
                                                 lidar=env["with_normals_path"],
                                                 output_matrix=env["matrix_path"],
                                                 max_distance=10)
            elif args.registration_method == "interactive":
                input("Get transformation matrix between {0} and {1} so that we should"
                      " apply it to the reconstructed point cloud to have the lidar point cloud, "
                      "and paste it in the file {2}. When done, press ENTER".format(env["with_normals_path"],
                                                                                    env["georefrecon_ply"],
                                                                                    env["matrix_path"]))
            env["global_registration_matrix"] = get_matrix(env["matrix_path"])
            mxw.apply_transform_to_project(env["lidar_mlp"], env["aligned_mlp"], env["global_registration_matrix"])
    else:
        env["global_registration_matrix"] = get_matrix(env["matrix_path"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Occlusion Mesh computing")
        if args.normals_method == "radius":
            eth3d.compute_normals(env["with_normals_path"], env["aligned_mlp"], neighbor_radius=args.normals_radius)
        else:
            eth3d.compute_normals(env["with_normals_path"], env["aligned_mlp"], neighbor_count=args.normals_neighbours)
        pcl_util.create_vis_file(env["georefrecon_ply"], env["with_normals_path"],
                                 resolution=args.mesh_resolution, output=env["with_normals_path"].stripext() + "_subsampled.ply")
        colmap.delaunay_mesh(env["occlusion_ply"], input_ply=env["with_normals_path"].stripext() + "_subsampled.ply")
        if args.splats:
            eth3d.create_splats(env["splats_ply"], env["with_normals_path"], env["occlusion_ply"], threshold=args.splat_threshold)

    if args.inspect_dataset:
        eth3d.inspect_dataset(scan_meshlab=env["aligned_mlp"],
                              colmap_model=env["georef_recon"],
                              image_path=env["image_path"])
        eth3d.inspect_dataset(scan_meshlab=env["aligned_mlp"],
                              colmap_model=env["georef_recon"],
                              image_path=env["image_path"],
                              occlusions=env["occlusion_ply"],
                              splats=env["splats_ply"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Groud Truth generation")
        for j, v in enumerate(env["videos_to_localize"]):
            video_env = env["videos_workspaces"][v]

            generate_GT(video_name=v, GT_already_done=video_env["GT_already_done"],
                        video_index=j+1,
                        num_videos=len(env["videos_to_localize"]),
                        metadata=video_env["metadata"],
                        **video_env["output_env"], **env)


if __name__ == '__main__':
    main()
