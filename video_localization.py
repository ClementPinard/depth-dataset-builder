import numpy as np
from path import Path

from cli_utils import print_step
import colmap_util.read_model as rm
from filter_colmap_model import filter_colmap_model
import pandas as pd
import add_video_to_db as avtd
import extract_pictures_from_model as epfm
import convert_dataset as cd
import generate_sky_masks as gsm
import meshlab_xml_writer as mxw
import resize_colmap_cameras as rcc


def is_video_in_model(video_name, colmap_model, metadata):

    mapped_images_ids = rm.read_images_binary(colmap_model/"images.bin").keys()
    video_image_ids = metadata["db_id"]
    return sum(video_image_ids.isin(mapped_images_ids)) > 0


def sift_and_match(colmap, more_sift_features, indexed_vocab_tree, image_list, max_num_matches, **env):
    tries = 0
    while tries < 10:
        try:
            colmap.extract_features(image_list=image_list, more=more_sift_features)
            colmap.match(method="sequential", vocab_tree=indexed_vocab_tree, max_num_matches=max_num_matches)
        except Exception:
            # If it failed, that's because sift gpu has failed.
            print("Error With colmap, will retry")
            tries += 1
            pass
        else:
            return


def error_empty():
    print("Error, empty localization")
    return
    # print("will try map from video")
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


def localize_video(video_name, video_frames_folder, thorough_db, metadata_path, lowfps_image_list_path, lowfps_db,
                   chunk_image_list_paths, chunk_dbs,
                   colmap_models_root, full_model, lowfps_model, chunk_models, final_model,
                   output_env, eth3d, colmap, ffmpeg, pcl_util,
                   step_index=None, video_index=None, num_videos=None, already_localized=False,
                   save_space=False, triangulate=False, **env):

    def print_step_pv(step_number, step_name):
        if step_index is not None and video_index is not None and num_videos is not None:
            progress = "{}/{}".format(video_index, num_videos)
            substep = "{}.{}".format(step_index, video_index)
        else:
            progress = ""
            substep = ""
        print_step("{}.{}".format(substep, step_number),
                   "[Video {}, {}] \n {}".format(video_name.basename(),
                                                 progress,
                                                 step_name))

    def clean_workspace():
        if save_space:
            with open(env["video_frame_list_thorough"], "r") as f:
                files_to_keep = [Path(path.split("\n")[0]) for path in f.readlines()]
            with open(lowfps_image_list_path, "r") as f:
                files_to_keep += [Path(path.split("\n")[0]) for path in f.readlines()]
            files_to_keep += [file.relpath(env["colmap_img_root"]) for file in [metadata_path,
                                                                                lowfps_image_list_path,
                                                                                *chunk_image_list_paths]]
            for file in sorted(video_frames_folder.files()):
                if file.relpath(env["colmap_img_root"]) not in files_to_keep:
                    file.remove()
            colmap_models_root.rmtree_p()

    # Perform checks if it has not already been computed
    if already_localized:
        print("already done")
        return

    i_pv = 1

    thorough_db.copy(lowfps_db)
    colmap.db = lowfps_db
    metadata = pd.read_csv(metadata_path)

    print_step_pv(i_pv, "Full video extraction")
    if save_space:
        ffmpeg.extract_images(video_name, video_frames_folder)
    else:
        print("Already Done.")

    i_pv += 1
    print_step_pv(i_pv, "Sky mask generation")
    gsm.process_folder(folder_to_process=video_frames_folder, **env)

    i_pv += 1
    print_step_pv(i_pv, "Complete photogrammetry with video at {} fps".format(env["lowfps"]))
    avtd.add_to_db(lowfps_db, metadata, lowfps_image_list_path)

    sift_and_match(colmap, image_list=lowfps_image_list_path, **env)

    lowfps_model.makedirs_p()
    colmap.map(output=lowfps_model, input=env["georef_recon"])
    if not is_video_in_model(video_name, lowfps_model, metadata):
        print("Error, video was not localized")
        error_empty()
        clean_workspace()
        return

    # when colmap map is called, the model is normalized so we have to georegister it again
    # Can be done either with model_aligner, or with model_merger
    # Additionally, we add the new positions to a full model that will be used for lidar registration
    # and also occlusion mesh computing
    colmap.merge_models(output=env["georef_full_recon"], input1=env["georef_full_recon"], input2=lowfps_model)
    colmap.merge_models(output=lowfps_model, input1=env["georef_recon"], input2=lowfps_model)
    # colmap.align_model(output=lowfps_model,
    #                    input=lowfps_model,
    #                    ref_images=env["georef_frames_list"])

    i_pv += 1
    print_step_pv(i_pv, "Localizing remaining frames")

    for k, (list_path, full_db, chunk_model) in enumerate(zip(chunk_image_list_paths,
                                                              chunk_dbs,
                                                              chunk_models)):
        print("\nLocalizing Chunk {}/{}".format(k + 1, len(chunk_dbs)))
        chunk_model.makedirs_p()
        lowfps_db.copy(full_db)
        colmap.db = full_db
        avtd.add_to_db(full_db, metadata, frame_list_path=list_path)
        sift_and_match(colmap, image_list=list_path, **env)
        colmap.register_images(output=chunk_model, input=lowfps_model)
        colmap.adjust_bundle(output=chunk_model, input=chunk_model)
    chunk_models[0].merge_tree(full_model)
    if len(chunk_model) > 1:
        for chunk in chunk_models[1:]:
            colmap.merge_models(output=full_model, input1=full_model, input2=chunk)
    final_model.makedirs_p()
    empty = not epfm.extract_pictures(input=full_model,
                                      output=final_model,
                                      picture_list=metadata["image_path"].values,
                                      output_format=".bin" if triangulate else ".txt")

    if empty:
        error_empty()
        clean_workspace()

    if triangulate:
        i_pv += 1
        print_step_pv(i_pv, "Re-Alignment of triangulated points with Lidar point cloud")

        colmap.triangulate_points(final_model, final_model)
        colmap.export_model(final_model, final_model, output_type="TXT")
        ply_name = final_model / "georef_{}.ply".format(video_name.stem)
        matrix_name = final_model / "matrix.txt"
        colmap.export_model(ply_name, final_model, output_type="PLY")
        pcl_util.register_reconstruction(georef=ply_name, lidar=env["lidar_ply"],
                                         output_matrix=matrix_name, output_cloud=env["lidar_ply"],
                                         max_distance=10)

    (final_model / "images.txt").rename(final_model / "images_raw.txt")
    (final_model / "cameras.txt").rename(final_model / "cameras_raw.txt")

    output_env["video_frames_folder"].makedirs_p()
    video_frames_folder.merge_tree(output_env["video_frames_folder"])

    output_env["model_folder"].makedirs_p()
    colmap_models_root.merge_tree(output_env["model_folder"])

    clean_workspace()


def generate_GT(video_name, raw_output_folder, images_root_folder, video_frames_folder,
                viz_folder, kitti_format_folder, metadata_path, interpolated_frames_list,
                final_model, aligned_mlp, global_registration_matrix,
                occlusion_ply, splats_ply,
                output_rescale, output_width,
                eth3d, colmap, filter_models=True,
                step_index=None, video_index=None, num_videos=None, GT_already_done=False,
                save_space=False, inspect_dataset=False, **env):

    def print_step_pv(step_number, step_name):
        if step_index is not None and video_index is not None and num_videos is not None:
            progress = "{}/{}".format(video_index, num_videos)
            substep = "{}.{}".format(step_index, video_index)
        else:
            progress = ""
            substep = ""
        print_step("{}.{}".format(substep, step_number),
                   "[Video {}, {}] \n {}".format(video_name.basename(),
                                                 progress,
                                                 step_name))

    if GT_already_done:
        return
    if not final_model.isdir():
        print("Video not localized, rerun the script without skipping former step")
        return

    print("Creating GT on video {} [{}/{}]".format(video_name.basename(), video_index, num_videos))
    i_pv = 1
    metadata = pd.read_csv(metadata_path).set_index("db_id", drop=False).sort_values("time")
    if filter_models:
        print_step_pv(i_pv, "Filtering model to have continuous localization")
        interpolated_frames = filter_colmap_model(input_images_colmap=final_model / "images_raw.txt",
                                                  output_images_colmap=final_model / "images.txt",
                                                  metadata=metadata, **env)
        with open(interpolated_frames_list, "w") as f:
            f.write("\n".join(interpolated_frames) + "\n")
        i_pv += 1
    else:
        (final_model / "images_raw.txt").copy(final_model / "images.txt")
        interpolated_frames = []

    rcc.resize_cameras(input_cameras=final_model / "cameras_raw.txt",
                       output_cameras=final_model / "cameras.txt",
                       output_width=output_width,
                       output_rescale=output_rescale)

    model_length = len(rm.read_images_text(final_model / "images.txt"))
    if model_length < 2:
        return

    '''
    In case the reconstructed model is only locally good, there's the possibility of having a specific
    transformation matrix per video in the final model folder, which might work better than the the global registration_matrix
    '''
    specific_matrix_path = final_model / "matrix.txt"
    if specific_matrix_path.isfile():
        registration_matrix = np.linalg.inv(np.fromfile(specific_matrix_path, sep=" ").reshape(4, 4))
        adjustment_matrix = registration_matrix * np.linalg.inv(global_registration_matrix)
        final_mlp = final_model / "aligned.mlp"
        final_occlusions = final_model / "occlusions.mlp"
        final_splats = final_model / "splats.mlp"
        mxw.apply_transform_to_project(aligned_mlp, final_mlp, adjustment_matrix)
        mxw.create_project(final_occlusions, [occlusion_ply], transforms=[adjustment_matrix])
        mxw.create_project(final_splats, [splats_ply], transforms=[adjustment_matrix])
        pose_scale = np.linalg.norm(adjustment_matrix[:, :3], 2)

    else:
        final_mlp = aligned_mlp
        final_occlusions = occlusion_ply
        final_splats = splats_ply
        pose_scale = np.linalg.norm(global_registration_matrix[:, :3], 2)

    if inspect_dataset:
        eth3d.image_path = images_root_folder
        # Do 3 inspections :
        #  - inspection with reconstructed cloud
        #  - inspection with lidar cloud without occlusion
        #  - inspection with lidar cloud and occlusion models
        # Careful, very RAM demanding for long sequences !
        print("THIRD DATASET INSPECTION")
        print("Inspection of localized video frames "
              "w.r.t Dense reconstruction")
        georef_mlp = env["georef_recon"]/"georef_recon.mlp"
        eth3d.inspect_dataset(georef_mlp, final_model)
        print("Inspection of localized video frames "
              "w.r.t Aligned Lidar Point Cloud")
        eth3d.inspect_dataset(final_mlp, final_model)
        print("Inspection of localized video frames "
              "w.r.t Aligned Lidar Point Cloud and Occlusion Meshes")
        eth3d.inspect_dataset(final_mlp, final_model,
                              final_occlusions, final_splats)

    i_pv += 1
    print_step_pv(i_pv, "Creating Ground truth data with ETH3D")

    eth3d.create_ground_truth(final_mlp, final_model, raw_output_folder,
                              final_occlusions, final_splats)
    viz_folder.makedirs_p()
    kitti_format_folder.makedirs_p()

    i_pv += 1
    print_step_pv(i_pv, "Convert to KITTI format and create video with GT visualization")

    cd.convert_dataset(final_model,
                       raw_output_folder / "ground_truth_depth" / video_name.stem,
                       images_root_folder,
                       raw_output_folder / "occlusion_depth" / video_name.stem,
                       kitti_format_folder, viz_folder, pose_scale=pose_scale,
                       metadata=metadata, interpolated_frames=interpolated_frames,
                       visualization=True, video=True, downscale=1, threads=8, **env)
    if filter_models:
        interpolated_frames_list.copy(kitti_format_folder)
    if save_space:
        (raw_output_folder / "occlusion_depth" / video_name.stem).rmtree_p()

    return


def generate_GT_individual_pictures(colmap_img_root, individual_pictures_list, raw_output_folder,
                                    converted_output_folder, input_colmap_model,
                                    aligned_mlp, relpath,
                                    occlusion_ply, splats_ply,
                                    eth3d, colmap, step_index=None,
                                    save_space=False, resume_work=False, **env):
    def print_step_pv(step_number, step_name):
        if step_index is not None:
            print_step("{}.{}".format(step_index, step_number), step_name)
        else:
            print_step(step_index, step_name)

    ground_truth_depth_folder = raw_output_folder / "ground_truth_depth" / relpath.stem
    occlusion_depth_folder = raw_output_folder / "occlusion_depth" / relpath.stem
    if resume_work and ground_truth_depth_folder.isdir():
        print("Directory {} already done, skipping...".format(relpath))
        return
    i_pv = 1
    print_step_pv(i_pv, "Copy individual images to output dataset {}".format(raw_output_folder))
    for p in individual_pictures_list:
        output_path = raw_output_folder / "images" / p
        output_path.parent.makedirs_p()
        (colmap_img_root / p).copy(output_path)

    i_pv += 1
    print_step_pv(i_pv, "Extract individual images to dedicated COLMAP model")
    pictures_colmap_model = raw_output_folder / "models" / relpath
    pictures_colmap_model.makedirs_p()
    epfm.extract_pictures(input=input_colmap_model,
                          output=pictures_colmap_model,
                          picture_list=individual_pictures_list,
                          output_format=".txt")

    i_pv += 1
    print_step_pv(i_pv, "Creating Ground truth data with ETH3D")
    eth3d.create_ground_truth(aligned_mlp,
                              pictures_colmap_model,
                              raw_output_folder,
                              occlusion_ply,
                              splats_ply)
    viz_folder = converted_output_folder / "visualization" / relpath
    viz_folder.makedirs_p()
    kitti_format_folder = converted_output_folder / "KITTI" / relpath
    kitti_format_folder.makedirs_p()

    i_pv += 1
    print_step_pv(i_pv, "Convert to KITTI format and create pictures with GT visualization")
    cd.convert_dataset(pictures_colmap_model,
                       ground_truth_depth_folder,
                       raw_output_folder / "images",
                       occlusion_depth_folder,
                       kitti_format_folder, viz_folder,
                       images_list=individual_pictures_list,
                       visualization=True, video=False, downscale=1, threads=8, **env)
    if save_space:
        (raw_output_folder / "occlusion_depth" / "individual_pictures").rmtree_p()
