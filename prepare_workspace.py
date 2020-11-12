def check_input_folder(path, with_lidar=True):
    def print_error_string():
        print("Error, bad input folder structure")
        print("Expected :")
        if with_lidar:
            print(str(path/"Lidar"))
        print(str(path/"Pictures"))
        print(str(path/"Videos"))
        print()
        print("but got :")
        print("\n".join(str(d) for d in path.dirs()))

    expected_folders = ["Pictures", "Videos"]
    if with_lidar:
        expected_folders.append("Lidar")
    if all((path/d).isdir() for d in expected_folders):
        return
    else:
        print_error_string()


def prepare_workspace(path, env, with_lidar=True):
    if with_lidar:
        env["lidar_path"] = path / "Lidar"
        env["lidar_mlp"] = env["workspace"] / "lidar.mlp"
        env["with_normals_path"] = env["lidar_path"] / "with_normals.ply"
        env["occlusion_ply"] = env["lidar_path"] / "occlusion_model.ply"
        env["splats_ply"] = env["lidar_path"] / "splats_model.ply" if env["splats"] else None
        env["occlusion_mlp"] = env["lidar_path"] / "occlusions.mlp"
        env["splats_mlp"] = env["lidar_path"] / "splats.mlp"
        env["matrix_path"] = env["workspace"] / "matrix_thorough.txt"
    else:
        env["occlusion_ply"] = path / "occlusion_model.ply"
        env["splats_ply"] = path / "splats_model.ply"

    env["image_path"] = path / "Pictures"
    env["mask_path"] = path / "Masks"
    env["video_path"] = path / "Pictures" / "Videos"
    env["thorough_recon"] = path / "Thorough"
    env["georef_recon"] = env["thorough_recon"] / "georef"
    env["georef_full_recon"] = env["thorough_recon"] / "georef_full"
    env["dense_workspace"] = env["thorough_recon"]/"dense"
    env["video_recon"] = path / "Videos_reconstructions"
    env["aligned_mlp"] = env["workspace"] / "aligned_model.mlp"

    env["centroid_path"] = path / "centroid.txt"
    env["thorough_db"] = path / "scan_thorough.db"
    env["video_frame_list_thorough"] = env["image_path"] / "video_frames_for_thorough_scan.txt"
    env["georef_frames_list"] = env["image_path"] / "georef.txt"

    env["georefrecon_ply"] = env["georef_recon"] / "georef_reconstruction.ply"
    env["indexed_vocab_tree"] = env["workspace"] / "vocab_tree_thorough.bin"


def prepare_video_workspace(video_name, video_frames_folder,
                            raw_output_folder, converted_output_folder,
                            video_recon, video_path, **env):
    video_env = {video_name: video_name,
                 video_frames_folder: video_frames_folder}
    relative_path_folder = video_frames_folder.relpath(video_path)
    video_env["lowfps_db"] = video_frames_folder / "video_low_fps.db"
    video_env["metadata"] = video_frames_folder / "metadata.csv"
    video_env["lowfps_image_list_path"] = video_frames_folder / "lowfps.txt"
    video_env["chunk_image_list_paths"] = sorted(video_frames_folder.files("full_chunk_*.txt"))
    video_env["chunk_dbs"] = [video_frames_folder / fp.stem + ".db" for fp in video_env["chunk_image_list_paths"]]
    colmap_root = video_recon / relative_path_folder
    video_env["colmap_models_root"] = colmap_root
    video_env["full_model"] = colmap_root
    video_env["lowfps_model"] = colmap_root / "lowfps"
    num_chunks = len(video_env["chunk_image_list_paths"])
    video_env["chunk_models"] = [colmap_root / "chunk_{}".format(index) for index in range(num_chunks)]
    video_env["final_model"] = colmap_root / "final"
    output = {}
    output["images_root_folder"] = raw_output_folder / "images"
    output["video_frames_folder"] = output["images_root_folder"] / "Videos" / relative_path_folder
    output["model_folder"] = raw_output_folder / "models" / relative_path_folder
    output["interpolated_frames_list"] = output["model_folder"] / "interpolated_frames.txt"
    output["final_model"] = output["model_folder"] / "final"
    output["kitti_format_folder"] = converted_output_folder / "KITTI" / relative_path_folder
    output["viz_folder"] = converted_output_folder / "visualization" / relative_path_folder
    video_env["output_env"] = output
    video_env["already_localized"] = env["resume_work"] and output["model_folder"].isdir()
    video_env["GT_already_done"] = env["resume_work"] and (raw_output_folder / "ground_truth_depth" / video_name.stem).isdir()
    return video_env
