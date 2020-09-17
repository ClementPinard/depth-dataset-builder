from pyproj import Proj
from edit_exif import get_gps_location
import numpy as np
import rawpy
import imageio
import generate_sky_masks as gsm
import videos_to_colmap as v2c
import colmap_util as ci


def extract_gps_and_path(existing_pictures, image_path, system, centroid=None, **env):
    proj = Proj(system)
    georef_list = []
    for img in existing_pictures:
        gps = get_gps_location(img)
        if gps is not None:
            lat, lng, alt = gps
            x, y = proj(lng, lat)
            if centroid is None:
                centroid = np.array([x, y, alt])
            x -= centroid[0]
            y -= centroid[1]
            alt -= centroid[2]
            georef_list.append("{} {} {} {}\n".format(img.relpath(image_path), x, y, alt))
    return georef_list, centroid


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


def extract_videos_to_workspace(video_path, video_frame_list_thorough, georef_frames_list, **env):
    existing_georef, env["centroid"] = extract_gps_and_path(**env)
    path_lists, extracted_video_folders = v2c.process_video_folder(output_video_folder=video_path, **env)
    if path_lists is not None:
        with open(video_frame_list_thorough, "w") as f:
            f.write("\n".join(path_lists["thorough"]["frames"]))
        with open(georef_frames_list, "w") as f:
            f.write("\n".join(existing_georef) + "\n")
            f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
        for v in env["videos_list"]:
            video_folder = extracted_video_folders[v]
            with open(video_folder / "lowfps.txt", "w") as f:
                f.write("\n".join(path_lists[v]["frames_lowfps"]) + "\n")
            with open(video_folder / "georef.txt", "w") as f:
                f.write("\n".join(existing_georef) + "\n")
                f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
                f.write("\n".join(path_lists[v]["georef_lowfps"]) + "\n")
            for j, l in enumerate(path_lists[v]["frames_full"]):
                with open(video_folder / "full_chunk_{}.txt".format(j), "w") as f:
                    f.write("\n".join(l) + "\n")
    gsm.process_folder(folder_to_process=video_path, **env)
    return extracted_video_folders


def choose_biggest_model(dir):
    colmap_model_dirs = dir.dirs("[0-9]*")
    model_sizes = [len(ci.read_model.read_images_binary(d/"images.bin")) for d in colmap_model_dirs]
    return colmap_model_dirs[model_sizes.index(max((model_sizes)))]
