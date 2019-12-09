import edit_exif
from subprocess import Popen, PIPE
from path import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from tqdm import tqdm


def extract_images(folder_path, file_path, fps):

    print("exporting to images with ffmpeg ...")
    if fps is not None:
        fps_arg = ["-vf", "fps={}".format(fps)]
    else:
        fps_arg = []

    ffmpeg = Popen(["ffmpeg", "-y", "-i", str(file_path), "-qscale:v", "2"] + fps_arg + [str(folder_path/"%05d.jpg")],
                   stdout=PIPE, stderr=PIPE)
    ffmpeg.wait()
    return sorted(folder_path.files("*.jpg"))


def extract_metadata(folder_path, file_path, native_wrapper):
    output_file = folder_path/"metadata.csv"
    print("extracting metadata with vmeta_extract...")
    vmeta_extract = Popen([native_wrapper, "vmeta-extract", str(file_path), "--csv", str(output_file)],
                          stdout=PIPE, stderr=PIPE)
    vmeta_extract.wait()
    return output_file


def add_gps_to_exif(csv_file, image_paths, fps):
    metadata = pd.read_csv(csv_file, sep=" ")
    metadata = metadata.set_index("time")
    metadata.index = pd.to_datetime(metadata.index, unit="us")

    if fps is not None:
        metadata = metadata.resample("{:.3f}S".format(1/fps)).first()
        metadata.to_csv(csv_file.stripext() + "_{}fps.csv".format(fps), sep=" ")

    print("Modifying gps EXIF for colmap...")
    for img_path, row in tqdm(zip(image_paths, metadata.iterrows()), total=len(image_paths)):
        if row[1]["location_valid"] == 1:
            edit_exif.set_gps_location(img_path,
                                       row[1]["location_latitude"],
                                       row[1]["location_longitude"],
                                       row[1]["location_altitude"])


def save_images_path_list(output_folder, origin, images_path_list):
    relative_path_lists = [origin.relpathto(img) + '\n' for img in images_path_list]
    with open(output_folder/'images.txt', 'w') as f:
        f.writelines(relative_path_lists)


def workflow(root, output_folder, video_path, args):
    print("Generating images with gps for video {}".format(str(video_path)))
    output_folder /= video_path.namebase
    if args.fps is not None:
        output_folder += "_{}fps".format(args.fps)
    output_folder.mkdir_p()
    images_path_list = extract_images(output_folder, video_path, args.fps)
    csv_path = extract_metadata(output_folder, video_path, args.nw)
    add_gps_to_exif(csv_path, images_path_list, args.fps)
    save_images_path_list(output_folder, args.origin or root, images_path_list)


parser = ArgumentParser(description='image extractor from parrot video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--input', metavar='PATH', default="~/Images/scan manoir/anafi/video",
                    help='path to video folder or video file', type=Path)
parser.add_argument('--fps', metavar='F', default=None, type=int,
                    help='fps')
parser.add_argument('--output_folder', metavar='DIR', default=None, type=Path)
parser.add_argument('--origin', metavar='DIR', default=None, type=Path,
                    help='folder relative to which the images path list will be generated')
parser.add_argument('--nw', default='',
                    help="native-wrapper.sh file location")

if __name__ == '__main__':
    file_exts = ['.mp4', '.MP4']
    args = parser.parse_args()
    if args.input.isfile() and args.input.ext in file_exts:
        root = args.input.parent
        videos = [args.input]
    elif args.input.isdir():
        root = args.input
        videos = sum([args.input.walkfiles('*{}'.format(ext)) for ext in file_exts], [])
        print("Found {} videos".format(len(videos)))

    if args.output_folder is None:
        args.output_folder = root
    for video_path in videos:
        workflow(root, args.output_folder/(video_path.parent.relpath(root)), video_path, args)
