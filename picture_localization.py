from wrappers import Colmap, ETH3D
from cli_utils import set_new_images_arparser, print_step, get_matrix
from video_localization import generate_GT_individual_pictures
import meshlab_xml_writer as mxw
import prepare_images as pi
import prepare_workspace as pw
import pcl_util
import numpy as np


def main():
    args = set_new_images_arparser().parse_args()
    env = vars(args)
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
    eth3d = ETH3D(args.eth3d, args.raw_output_folder / "Images", args.max_occlusion_depth,
                  verbose=args.verbose, logfile=args.log, splat_radius=args.eth3d_splat_radius)
    env["eth3d"] = eth3d
    env["videos_list"] = sum((list((args.input_folder/"Videos").walkfiles('*{}'.format(ext))) for ext in args.vid_ext), [])
    no_gt_folder = args.input_folder/"Videos"/"no_groundtruth"
    if no_gt_folder.isdir():
        env["videos_to_localize"] = [v for v in env["videos_list"] if not str(v).startswith(no_gt_folder)]

    i = 1
    print_step(i, "Pictures preparation")
    env["individual_pictures"] = pi.extract_pictures_to_workspace(**env)

    i += 1
    print_step(i, "Add new pictures to COLMAP thorough model")
    colmap.db = env["thorough_db"]
    colmap.match(method="vocab_tree", vocab_tree=env["indexed_vocab_tree"], max_num_matches=env["max_num_matches"])
    extended_georef = env["georef_recon"] + "_extended"
    extended_georef.makedirs_p()
    if args.map_new_images:
        colmap.map(input=env["georef_recon"], output=extended_georef)
    else:
        colmap.register_images(input=env["georef_recon"], output=extended_georef)
    colmap.adjust_bundle(extended_georef, extended_georef,
                         num_iter=args.bundle_adjuster_steps, refine_extra_params=True)
    colmap.merge_models(output=env["georef_full_recon"], input1=env["georef_full_recon"], input2=extended_georef)

    if env["rebuild_occlusion_mesh"]:
        i += 1
        print_step(i, "Full reconstruction point cloud densificitation with new images")
        colmap.undistort(input=env["georef_full_recon"])
        # This step should be fast since everything else than new images is already computed
        colmap.dense_stereo(min_depth=env["stereo_min_depth"], max_depth=env["stereo_max_depth"])
        colmap.stereo_fusion(output=env["georefrecon_ply"])
        if args.inspect_dataset:
            georef_mlp = env["georef_full_recon"]/"georef_recon.mlp"
            mxw.create_project(georef_mlp, [env["georefrecon_ply"]])
            colmap.export_model(output=env["georef_full_recon"],
                                input=env["georef_full_recon"],
                                output_type="TXT")
            eth3d.inspect_dataset(scan_meshlab=georef_mlp,
                                  colmap_model=env["georef_full_recon"],
                                  image_path=env["colmap_img_root"])
            eth3d.inspect_dataset(scan_meshlab=env["aligned_mlp"],
                                  colmap_model=env["georef_full_recon"],
                                  image_path=env["colmap_img_root"])

        i += 1
        print_step(i, "Occlusion Mesh re-computing")
        '''combine the MLP file into a single ply file. We need the normals for the splats'''
        if args.normals_method == "radius":
            eth3d.compute_normals(env["with_normals_path"], env["aligned_mlp"], neighbor_radius=args.normals_radius)
        else:
            eth3d.compute_normals(env["with_normals_path"], env["aligned_mlp"], neighbor_count=args.normals_neighbours)
        '''Create vis file that will tell by what images each point can be seen. We transfer this knowledge from georefrecon
        to the Lidar model'''
        env["global_registration_matrix"] = get_matrix(env["matrix_path"])
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

    i += 1
    if i not in args.skip_step:
        print_step(i, "Groud Truth generation")
        by_folder = pi.group_pics_by_folder(env["individual_pictures"])
        for folder, pic_list in by_folder.items():
            generate_GT_individual_pictures(input_colmap_model=env["georef_full_recon"],
                                            individual_pictures=pic_list,
                                            relpath=folder,
                                            step_index=i, **env)


if __name__ == '__main__':
    main()
