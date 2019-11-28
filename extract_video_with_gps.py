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


def extract_metadata(folder_path, file_path, native_wrapper):
    output_file = folder_path/"metadata.csv"
    print("extracting metadata with vmeta_extract...")
    vmeta_extract = Popen([native_wrapper, "vmeta-extract", str(file_path), "--csv", str(output_file)],
                          stdout=PIPE, stderr=PIPE)
    vmeta_extract.wait()


def add_gps_to_exif(folder, fps):
    csv_file = folder/"metadata.csv"
    metadata = pd.read_csv(csv_file, sep=" ")
    metadata = metadata.set_index("time")
    metadata.index = pd.to_datetime(metadata.index, unit="us")

    if fps is not None:
        metadata = metadata.resample("{:.3f}S".format(1/fps)).first()

    pictures = sorted(folder.files("*.jpg"))

    print("Modifying gps EXIF for colmap...")
    for pic_path, row in tqdm(zip(pictures, metadata.iterrows()), total=len(pictures)):
        if row[1]["location_valid"] == 1:
            edit_exif.set_gps_location(pic_path,
                                       row[1]["location_latitude"],
                                       row[1]["location_longitude"],
                                       row[1]["location_altitude"])


def workflow(output_folder, video_path, args):
    output_folder /= video_path.namebase
    if args.fps is not None:
        output_folder += "_{}fps".format(args.fps)
    output_folder.mkdir_p()
    print(video_path.namebase)
    print(video_path)
    extract_images(output_folder, video_path, args.fps)
    extract_metadata(output_folder, video_path, args.nw)
    add_gps_to_exif(output_folder, args.fps)


parser = ArgumentParser(description='image extractor from parrot video',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--root', metavar='DIR', default="~/Images/scan manoir/anafi/video",
                    help='path to video folder root', type=Path)
parser.add_argument('--fps', metavar='F', default=None, type=int,
                    help='fps')
parser.add_argument('--output_folder', metavar='PATH', default=None, type=Path)
parser.add_argument('--nw', default='',
                    help="native-wrapper.sh file location")

if __name__ == '__main__':
    args = parser.parse_args()
    root = args.root
    if args.output_folder is None:
        output_folder = root
    else:
        output_folder = args.output_folder
    file_exts = ['.mp4', '.MP4']
    if root.isdir():
        folders = list(root.walkdirs())
        folders.append(root)
        for folder in folders:
            videos = sum((folder.files('*{}'.format(ext)) for ext in file_exts), [])
            if videos:
                print("Generating images with gps for videos in {}".format(str(folder)))
                for video_path in videos:
                    workflow(output_folder/(folder.relpath(root)), video_path, args)
    elif root.isfile() and root.ext in file_exts:
        workflow(root.parent, output_folder, args)
