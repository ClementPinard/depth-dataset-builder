from pyproj import Proj
from edit_exif import get_gps_location
import numpy as np
import pandas as pd
import rawpy
import imageio
import generate_sky_masks as gsm
import videos_to_colmap as v2c
import colmap_util as ci


def extract_gps_and_path(individual_pictures, colmap_img_root, system, centroid=None, **env):
    proj = Proj(system)
    georef_list = []
    for img in individual_pictures:
        gps = get_gps_location(colmap_img_root / img)
        if gps is not None:
            lat, lng, alt = gps
            x, y = proj(lng, lat)
            if centroid is None:
                centroid = np.array([x, y, alt])
            x -= centroid[0]
            y -= centroid[1]
            alt -= centroid[2]
            georef_list.append("{} {} {} {}\n".format(img, x, y, alt))
    return georef_list, centroid


def extract_pictures_to_workspace(input_folder, colmap_img_root, individual_pictures_path, workspace, colmap,
                                  raw_ext, pic_ext, more_sift_features, generic_model, **env):
    picture_folder = input_folder / "Pictures"
    pictures = []
    raw_files = sum((list(picture_folder.walkfiles('*{}'.format(ext))) for ext in raw_ext), [])
    for raw in raw_files:
        if not any((raw.stripext() + ext).isfile() for ext in pic_ext):
            converted_pic_path = raw.relpath(picture_folder).stripext() + '.jpg'
            if not converted_pic_path.isfile():
                raw_array = rawpy.imread(raw)
                rgb = raw_array.postprocess()
                dst = individual_pictures_path / converted_pic_path
                dst.parent.makedirs_p()
                imageio.imsave(dst, rgb)
            pictures.append(converted_pic_path)
    normal_files = sum((list(picture_folder.walkfiles('*{}'.format(ext))) for ext in pic_ext), [])
    for file in normal_files:
        pic_path = file.relpath(picture_folder)
        if not pic_path.isfile():
            dst = individual_pictures_path / pic_path
            dst.parent.makedirs_p()
            file.copy(dst)
        pictures.append(colmap_img_root.relpathto(individual_pictures_path) / pic_path)
    gsm.process_folder(folder_to_process=individual_pictures_path, colmap_img_root=colmap_img_root, pic_ext=pic_ext, **env)
    with open(picture_folder / "individual_pictures.txt", 'w') as f:
        f.write("\n".join(pictures) + "\n")
    colmap.extract_features(per_sub_folder=True, model=generic_model,
                            image_list=picture_folder/"individual_pictures.txt", more=more_sift_features)
    return pictures


def extract_videos_to_workspace(video_path, video_frame_list_thorough, georef_frames_list, **env):
    existing_georef, env["centroid"] = extract_gps_and_path(**env)
    if env["full_metadata"] is None:
        env["full_metadata"] = env["workspace"] / "full_metadata.csv"
    if env["full_metadata"].isfile():
        existing_metadata = pd.read_csv(env["full_metadata"])
    else:
        existing_metadata = None
    path_lists, extracted_video_folders, full_metadata = v2c.process_video_folder(output_video_folder=video_path,
                                                                                  existing_georef=existing_georef,
                                                                                  existing_metadata=existing_metadata,
                                                                                  **env)
    if path_lists is not None:
        full_metadata.to_csv(env["full_metadata"])
        with open(video_frame_list_thorough, "w") as f:
            f.write("\n".join(path_lists["thorough"]["frames"]))
        with open(georef_frames_list, "w") as f:
            f.write("\n".join(existing_georef) + "\n")
            f.write("\n".join(path_lists["thorough"]["georef"]) + "\n")
        for v, video_folder in extracted_video_folders.items():
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


def group_pics_by_folder(pictures):
    result = {}
    for p in pictures:
        key = p.parent
        if p.parent not in result.keys():
            result[key] = [p]
        else:
            result[key].append(p)
    return result
