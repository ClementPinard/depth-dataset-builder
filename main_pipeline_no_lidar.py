import numpy as np
from wrappers import Colmap, FFMpeg, PDraw, ETH3D, PCLUtil
from cli_utils import set_argparser, print_step, print_workflow
from video_localization import localize_video, generate_GT
import meshlab_xml_writer as mxw
import prepare_images as pi
import prepare_workspace as pw


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
    pw.check_input_folder(args.input_folder, with_lidar=False)
    args.workspace = args.workspace.abspath()
    pw.prepare_workspace(args.workspace, env, with_lidar=False)
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
    env["videos_list"] = sum((list((args.input_folder/"Videos").walkfiles('*{}'.format(ext))) for ext in args.vid_ext), [])
    no_gt_folder = args.input_folder/"Videos"/"no_groundtruth"
    if no_gt_folder.isdir():
        env["videos_to_localize"] = [v for v in env["videos_list"] if not str(v).startswith(no_gt_folder)]
    else:
        env["videos_to_localize"] = env["videos_list"]

    i = 1
    if i not in args.skip_step:
        print_step(i, "Pictures preparation")
        env["existing_pictures"] = pi.extract_pictures_to_workspace(**env)
    else:
        env["existing_pictures"] = sum((list(env["image_path"].walkfiles('*{}'.format(ext))) for ext in env["pic_ext"]), [])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Extracting Videos and selecting optimal frames for a thorough scan")
        env["videos_frames_folders"] = pi.extract_videos_to_workspace(fps=args.lowfps, **env)
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
        colmap.match(method="vocab_tree", vocab_tree=env["indexed_vocab_tree"])
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
                           input=env["thorough_recon"] / "0",
                           ref_images=env["georef_frames_list"])
        if not (env["georef_frames_list"]/"images.bin").isfile():
            # GPS alignment failed, possibly because not enough GPS referenced images
            # Copy the original model without alignment
            (env["thorough_recon"] / "0").merge_tree(env["georef_full_recon"])
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
                           step_index=i, video_index=j+1,
                           num_videos=len(env["videos_to_localize"]),
                           **video_env, **env)

    i += 1
    if i not in args.skip_step:
        print_step(i, "Full reconstruction point cloud densificitation")
        env["georef_full_recon"].makedirs_p()
        colmap.undistort(input=env["georef_full_recon"])
        colmap.dense_stereo()
        colmap.stereo_fusion(output=env["georefrecon_ply"])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Reconstruction cleaning")
        filtered = env["georefrecon_ply"].stripext() + "_filtered.ply"
        pcl_util.filter_cloud(input_file=env["georefrecon_ply"],
                              output_file=filtered,
                              knn=args.SOR[0], std=args.SOR[1])
        mxw.create_project(env["aligned_mlp"], [filtered])

    i += 1
    if i not in args.skip_step:
        print_step(i, "Occlusion Mesh computing")
        colmap.delaunay_mesh(env["occlusion_ply"], input_ply=env["georefrecon_ply"])
        if args.splats:
            eth3d.create_splats(env["splats_ply"], env["georefrecon_ply"].stripext() + "_filtered.ply", env["occlusion_ply"], threshold=args.splat_threshold)

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
                        global_registration_matrix=np.eye(4),
                        **video_env["output_env"], **env)


if __name__ == '__main__':
    main()
