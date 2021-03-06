from subprocess import Popen, PIPE
from .default_wrapper import Wrapper
import json


class FFMpeg(Wrapper):
    def __init__(self, binary="ffmpeg", probe="ffprobe", *args, **kwargs):
        super().__init__(binary, *args, **kwargs)
        self.probe = probe

    def extract_images(self, video_file, output_folder, fps=None):
        if fps is not None:
            fps_arg = ["-vf", "fps={}".format(fps)]
        else:
            fps_arg = []

        self.__call__(["-y", "-i", str(video_file), "-vsync", "0", "-qscale:v", "2"]
                      + fps_arg + [str(output_folder / output_folder.stem + "_%05d.jpg")])
        return sorted(output_folder.files("*.jpg"))

    def extract_specific_frames(self, video_file, output_folder, frame_ids):
        '''
        Typical command string :
        ffmpeg -i in.mp4 -vf select='eq(n\\,100)+eq(n\\,184)+eq(n\\,213)' -vsync 0 frames%d.jpg

        Note: Surprisingly, frames in the eq function are starting at 0, whereas the rest is starting at 1,
        so we need to decrement the frame index compared to what we would have got when extracting every frame
        '''
        select_string = "select='" + '+'.join(['eq(n\\,{})'.format(f-1) for f in frame_ids]) + "'"
        frame_string = output_folder/(video_file.stem + "tmp_%05d.jpg")
        ffmpeg_options = ["-y", "-i", video_file,
                          "-vf", select_string, "-vsync", "0",
                          "-qscale:v", "2", frame_string]
        self.__call__(ffmpeg_options)
        frame_files = sorted(output_folder.files(video_file.stem + "tmp_*.jpg"))
        assert(len(frame_files) == len(frame_ids)), \
            "error, got {} frame_ids, but got {} images extracted".format(len(frame_ids), len(frame_files))

        for f, frame_id in zip(frame_files, frame_ids):
            f.rename(f.parent / (video_file.stem + "_{:05d}.jpg".format(frame_id)))
        return sorted(output_folder.files("*.jpg"))

    def get_size_and_framerate(self, video_file):
        probe_process = Popen([self.probe, "-show_entries", "stream=height,width,r_frame_rate,nb_frames",
                               "-of", "json", "-select_streams", "v", str(video_file)],
                              stdout=PIPE, stderr=PIPE)
        json_cam = json.loads(probe_process.communicate()[0])['streams']
        if len(json_cam) > 1:
            print("Warning for video {0} : Multiple streams detected ({1}), only the first one will be considered, "
                  "please split the file into {1} separate mp4 files to analyze everything".format(video_file.basename(), len(json_cam)))
        return (int(json_cam[0]["width"]),
                int(json_cam[0]["height"]),
                frac_to_float(json_cam[0]["r_frame_rate"]),
                int(json_cam[0]["nb_frames"]))

    def create_video(self, video_path, input_string, glob=True, fps=30):
        ffmpeg_options = ["-y", "-r", "{:.2f}".format(fps)] + \
                         (["-pattern_type", "glob"] if glob else []) + \
                         ["-i", input_string, video_path]
        self.__call__(ffmpeg_options)


def frac_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        return float(num) / float(denom)
