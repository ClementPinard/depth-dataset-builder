import yaml
from path import Path


def add_global_options(parser, config_path=None):
    if config_path is not None:
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
    else:
        cfg = {}
    exec_paths = cfg.get("exec_path", {})
    parser.add_argument('--nw', default=exec_paths.get("native-wrapper", Path('')),
                        help="native-wrapper.sh file location", type=Path)
    parser.add_argument("--colmap", default=exec_paths.get("colmap", "colmap"),
                        type=Path, help="colmap exec file location")
    parser.add_argument("--eth3d", default=exec_paths.get("eth3d", Path("../dataset-pipeline/build")),
                        type=Path, help="ETHD3D detaset pipeline exec files folder location")
    parser.add_argument("--ffmpeg", default=exec_paths.get("ffmpeg", "ffmpeg"))
