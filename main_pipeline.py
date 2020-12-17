import las2ply
import numpy as np
from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from cli_utils import set_full_argparser, print_step, print_workflow, get_matrix
from video_localization import localize_video, generate_GT, generate_GT_individual_pictures
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
        if pointcloud_resolution is not None:
            pcl_util.subsample(input_file=ply.stripext() + "_filtered.ply",
                               output_file=ply.stripext() + "_subsampled.ply",
                               resolution=pointcloud_resolution)
            converted_clouds.append(ply.stripext() + "_subsampled.ply")
        else:
            converted_clouds.append(ply.stripext() + "_filtered.ply")
    temp_mlp = env["workspace"] / "lidar_unaligned.mlp"
    mxw.create_project(temp_mlp, converted_clouds, labels=None, transforms=None)
    if len(converted_clouds) > 1:
        eth3d.align_with_ICP(temp_mlp, env["lidar_mlp"], scales=5)
    else:
        temp_mlp.move(env["lidar_mlp"])

    return converted_clouds, output_centroid


def main():
    args = set_full_argparser().parse_args()
    env = vars(args)
    if args.show_steps:
        print_workflow()
        return
    if args.add_new_videos:
        env["resume_work"] = True
        args.skip_step = [1, 2, 4, 5, 8]
    if args.begin_step is not None:
        args.skip_step += list(range(args.begin_step))
    pw.check_input_folder(args.input_folder)
    args.workspace = args.workspace.abspath()
    pw.prepare_workspace(args.workspace, env)
    colmap = Colmap(db=env["thorough_db"],
                    image_path=env["colmap_img_root"],
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
                  verbose=args.verbose, logfile=args.log, splat_radius=args.eth3d_splat_radius)
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
        env["individual_pictures"] = pi.extract_pictures_to_workspace(**env)
    else:
        full_paths = sum((list(env["individual_pictures_path"].walkfiles('*{}'.format(ext))) for ext in env["pic_ext"]), [])
        env["individual_pictures"] = [path.relpath(env["colmap_img_root"]) for path in full_paths]

    i += 1
    # Get already existing_videos
    env["videos_frames_folders"] = {}
    by_name = {v.stem: v for v in env["videos_list"]}
    for folder in env["video_path"].walkdirs():
        video_name = folder.basename()
        if video_name in by_name.keys():
            env["videos_frames_folders"][by_name[video_name]] = folder
    if i not in args.skip_step:
        print_step(i, "Extracting Videos and selecting optimal frames for a thorough scan")
        new_video_frame_folders = pi.extract_videos_to_workspace(fps=args.lowfps, **env)
        # Concatenate both already treated videos and newly detected videos
        env["videos_frames_folders"] = {**env["videos_frames_folders"], **new_video_frame_folders}
    env["videos_workspaces"] = {}
    for v, frames_folder in env["videos_frames_folders"].items():
        env["videos_workspaces"][v] = pw.prepare_video_workspace(v, frames_folder, **env)

    i += 1
    if i not in args.skip_step:
        print_step(i, "First thorough photogrammetry")
        env["thorough_recon"].makedirs_p()
        colmap.extract_features(image_list=env["video_frame_list_thorough"], more=args.more_sift_features)
        colmap.index_images(vocab_tree_output=env["indexed_vocab_tree"], vocab_tree_input=args.vocab_tree)
        if env["match_method"] == "vocab_tree":
            colmap.match(method="vocab_tree", vocab_tree=env["indexed_vocab_tree"], max_num_matches=env["max_num_matches"])
        else:
            colmap.match(method="exhaustive", max_num_matches=env["max_num_matches"])
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
        env["georef_full_recon"].makedirs_p()
        colmap.align_model(output=env["georef_recon"],
                           input=thorough_model,
                           ref_images=env["georef_frames_list"])
        if not (env["georef_recon"]/"images.bin").isfile():
            # GPS alignment failed, possibly because not enough GPS referenced images
            # Copy the original model without alignment
            print("Warning, model alignment failed, the model will be normalized, and thus the depth maps too")
            thorough_model.merge_tree(env["georef_recon"])
        env["georef_recon"].merge_tree(env["georef_full_recon"])
    if args.inspect_dataset:
        print("FIRST DATASET INSPECTION")
        print("Inspection of localisalization of frames used in thorough mapping "
              "w.r.t Sparse reconstruction")
        colmap.export_model(output=env["georef_recon"] / "georef_sparse.ply",
                            input=env["georef_recon"])
        georef_mlp = env["georef_recon"]/"georef_recon.mlp"
        mxw.create_project(georef_mlp, [env["georef_recon"] / "georef_sparse.ply"])
        colmap.export_model(output=env["georef_recon"],
                            input=env["georef_recon"],
                            output_type="TXT")
        eth3d.inspect_dataset(scan_meshlab=georef_mlp,
                              colmap_model=env["georef_recon"],
                              image_path=env["colmap_img_root"])

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
        colmap.undistort(input=env["georef_full_recon"])
        colmap.dense_stereo(min_depth=env["stereo_min_depth"], max_depth=env["stereo_max_depth"])
        colmap.stereo_fusion(output=env["georefrecon_ply"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Alignment of photogrammetric reconstruction with respect to Lidar Point Cloud")
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
        mxw.apply_transform_to_project(env["lidar_mlp"], env["aligned_mlp"], env["global_registration_matrix"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Occlusion Mesh computing")
        '''combine the MLP file into a single ply file. We need the normals for the splats'''
        if args.normals_method == "radius":
            eth3d.compute_normals(env["with_normals_path"], env["aligned_mlp"], neighbor_radius=args.normals_radius)
        else:
            eth3d.compute_normals(env["with_normals_path"], env["aligned_mlp"], neighbor_count=args.normals_neighbours)
        '''Create vis file that will tell by what images each point can be seen. We transfer this knowledge from georefrecon
        to the Lidar model'''
        scale = np.linalg.norm(env["global_registration_matrix"][:3, :3], ord=2)
        with_normals_subsampled = env["with_normals_path"].stripext() + "_subsampled.ply"
        pcl_util.create_vis_file(env["georefrecon_ply"], env["with_normals_path"],
                                 resolution=args.mesh_resolution / scale,
                                 output=with_normals_subsampled)
        '''Compute the occlusion mesh by fooling COLMAP into thinking the lidar point cloud was made with colmap'''
        colmap.delaunay_mesh(env["occlusion_ply"], input_ply=with_normals_subsampled)
        if args.splats:
            eth3d.create_splats(env["splats_ply"], with_normals_subsampled,
                                env["occlusion_ply"], env["splat_threshold"] / scale,
                                env["max_splat_size"])

    if args.inspect_dataset:
        # First inspection : Check registration of the Lidar pointcloud wrt to COLMAP model but without the occlusion mesh
        # Second inspection : Check the occlusion mesh and the splats
        georef_mlp = env["georef_recon"]/"georef_recon.mlp"
        mxw.create_project(georef_mlp, [env["georefrecon_ply"]])
        colmap.export_model(output=env["georef_full_recon"],
                            input=env["georef_full_recon"],
                            output_type="TXT")
        print("SECOND DATASET INSPECTION")
        print("Inspection of localisalization of frames used in thorough mapping "
              "w.r.t Dense reconstruction")
        eth3d.inspect_dataset(scan_meshlab=georef_mlp,
                              colmap_model=env["georef_full_recon"],
                              image_path=env["colmap_img_root"])
        print("Inspection of localisalization of frames used in thorough mapping "
              "w.r.t Aligned Lidar Point Cloud")
        eth3d.inspect_dataset(scan_meshlab=env["aligned_mlp"],
                              colmap_model=env["georef_full_recon"],
                              image_path=env["colmap_img_root"])
        print("Inspection of localisalization of frames used in thorough mapping "
              "w.r.t Aligned Lidar Point Cloud and Occlusion Meshes")
        eth3d.inspect_dataset(scan_meshlab=env["aligned_mlp"],
                              colmap_model=env["georef_full_recon"],
                              image_path=env["colmap_img_root"],
                              occlusions=env["occlusion_ply"],
                              splats=env["splats_ply"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Groud Truth generation")
        for j, v in enumerate(env["videos_to_localize"]):
            video_env = env["videos_workspaces"][v]

            generate_GT(video_name=v, GT_already_done=video_env["GT_already_done"],
                        video_index=j+1,
                        step_index=i,
                        num_videos=len(env["videos_to_localize"]),
                        metadata_path=video_env["metadata_path"],
                        **video_env["output_env"], **env)
        if env["generate_groundtruth_for_individual_images"]:
            by_folder = pi.group_pics_by_folder(env["individual_pictures"])
            for folder, pic_list in by_folder.items():
                generate_GT_individual_pictures(input_colmap_model=env["georef_full_recon"],
                                                individual_pictures_list=pic_list,
                                                relpath=folder,
                                                step_index=i, **env)


if __name__ == '__main__':
    main()
