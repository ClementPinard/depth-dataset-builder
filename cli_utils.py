from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path


def set_argparser():
    parser = ArgumentParser(description='Main pipeline, from LIDAR pictures and videos to GT depth enabled videos',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    main_parser = parser.add_argument_group("Main options")
    main_parser.add_argument('--input_folder', metavar='PATH', default=Path("."), type=Path,
                             help="Folder with LAS point cloud, videos, and images")
    main_parser.add_argument('--workspace', metavar='PATH', default=Path("."),
                             help='path to workspace where COLMAP operations will be done', type=Path)
    main_parser.add_argument('--raw_output_folder', metavar='PATH', default=Path("."),
                             help='path to output folder : must be big !', type=Path)
    main_parser.add_argument('--converted_output_folder', metavar='PATH', default=Path("."),
                             help='path to output folder : must be big !', type=Path)

    main_parser.add_argument('--skip_step', metavar="N", nargs="*", default=[], type=int,
                             help='Skip selected steps')
    main_parser.add_argument('--begin_step', metavar="N", type=int, default=None)
    main_parser.add_argument('--show_steps', action="store_true")
    main_parser.add_argument('-v', '--verbose', action="count", default=0)
    main_parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"],
                             help='Video extensions to scrape from input folder')
    main_parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"],
                             help='Image extensions to scrape from input folder')
    main_parser.add_argument('--raw_ext', nargs='+', default=[".ARW", ".NEF", ".DNG"],
                             help='Raw Image extensions to scrape from input folder')
    main_parser.add_argument('--resume_work', action="store_true",
                             help='If selected, will try to skip video aready localized, and ground truth already generated')
    main_parser.add_argument('--inspect_dataset', action="store_true",
                             help='If selected, will open a window to inspect the dataset. '
                                  'See https://github.com/ETH3D/dataset-pipeline#dataset-inspection')
    main_parser.add_argument('--registration_method', choices=["simple", "eth3d", "interactive"], default="simple",
                             help='Method used for point cloud registration. See README, Manual step by step : step 11')

    pcp_parser = parser.add_argument_group("PointCLoud preparation")
    pcp_parser.add_argument("--pointcloud_resolution", default=None, type=float,
                            help='If set, will subsample the Lidar point clouds at the chosen resolution')
    pcp_parser.add_argument("--SOR", default=[10, 6], nargs=2, type=float,
                            help="Satistical Outlier Removal parameters : Number of nearest neighbours, max relative distance to standard deviation")

    ve_parser = parser.add_argument_group("Video extractor")
    ve_parser.add_argument('--total_frames', default=500, type=int)
    ve_parser.add_argument('--orientation_weight', default=1, type=float,
                           help="Weight applied to orientation during optimal sample. "
                           "Higher means two pictures with same location but different "
                           "orientation will be considered farer apart")
    ve_parser.add_argument('--resolution_weight', default=1, type=float,
                           help="same as orientation, but with image size")
    ve_parser.add_argument('--num_neighbours', default=10, type=int,
                           help='Number of frame shared between subsequent chunks')
    ve_parser.add_argument('--system', default="epsg:2154",
                           help='coordinates system used for GPS, should be the same as the LAS files used')
    ve_parser.add_argument('--lowfps', default=1, type=int,
                           help="framerate at which videos will be scanned WITH reconstruction")
    ve_parser.add_argument('--max_sequence_length', default=4000, type=int,
                           help='Number max of frames for a chunk. '
                           'This is for RAM purpose, as loading feature matches of thousands of frames can take up GBs of RAM')
    ve_parser.add_argument('--include_lowfps_thorough', action='store_true',
                           help="if selected, will include videos frames at lowfps for thorough scan (longer)")

    exec_parser = parser.add_argument_group("Executable files")
    exec_parser.add_argument('--log', default=None, type=Path)
    exec_parser.add_argument('--nw', default="native-wrapper.sh", type=Path,
                             help="native-wrapper.sh file location (see Anafi SDK documentation)")
    exec_parser.add_argument("--colmap", default="colmap", type=Path,
                             help="colmap exec file location")
    exec_parser.add_argument("--eth3d", default="../dataset-pipeline/build",
                             type=Path, help="ETH3D detaset pipeline exec files folder location")
    exec_parser.add_argument("--ffmpeg", default="ffmpeg", type=Path)
    exec_parser.add_argument("--pcl_util", default="pcl_util/build", type=Path)

    pm_parser = parser.add_argument_group("Photogrammetry")
    pm_parser.add_argument('--max_num_matches', default=32768, type=int, help="max number of matches, lower it if you get GPU memory error")
    pm_parser.add_argument('--vocab_tree', type=Path, default="vocab_tree_flickr100K_words256K.bin")
    pm_parser.add_argument('--triangulate', action="store_true")
    pm_parser.add_argument('--multiple_models', action='store_true', help='If selected, will let colmap mapper do multiple models.'
                                                                          'The biggest one will then be chosen')
    pm_parser.add_argument('--more_sift_features', action="store_true")
    pm_parser.add_argument('--save_space', action="store_true")
    pm_parser.add_argument('--add_new_videos', action="store_true")
    pm_parser.add_argument('--stereo_min_depth', type=float, default=0.1)
    pm_parser.add_argument('--stereo_max_depth', type=float, default=100)

    om_parser = parser.add_argument_group("Occlusion Mesh")
    om_parser.add_argument('--normals_method', default="radius", choices=["radius", "neighbours"])
    om_parser.add_argument('--normals_radius', default=0.2, type=float)
    om_parser.add_argument('--normals_neighbours', default=8, type=int)
    om_parser.add_argument('--mesh_resolution', default=0.2, type=float)
    om_parser.add_argument('--splats', action='store_true')
    om_parser.add_argument('--splat_threshold', default=0.1, type=float)
    om_parser.add_argument('--max_occlusion_depth', default=250, type=float)
    return parser


def print_step(step_number, step_name):
    print("\n\n=================")
    print("Step {}".format(step_number))
    print(step_name)
    print("=================")


global_steps = ["Point Cloud Preparation",
                "Pictures preparation",
                "Extracting Videos and selecting optimal frames for a thorough scan",
                "First thorough photogrammetry",
                "Alignment of photogrammetric reconstruction with GPS",
                "Video localization with respect to reconstruction",
                "Full reconstruction point cloud densificitation",
                "Alignment of photogrammetric reconstruction with Lidar point cloud",
                "Occlusion Mesh computing",
                "Ground Truth creation"]

per_vid_steps_1 = ["Full video extraction",
                   "Sky mask generation",
                   "Complete photogrammetry with video at 1 fps",
                   "Localizing remaining frames",
                   "Re-Alignment of triangulated points with Lidar point cloud"]
per_vid_steps_2 = ["Creating Ground truth data",
                   "Create video with GT vizualisation",
                   "Convert to KITTI format"]


def print_workflow():
    print("Global steps :")
    for i, s in enumerate(global_steps):
        print("{}) {}".format(i+1, s))
        if i == 4:
            print("\tPer video:")
            for i, s in enumerate(per_vid_steps_1):
                print("\t{}) {}".format(i+1, s))

    print("\tPer video:")
    for i, s in enumerate(per_vid_steps_2):
        print("\t{}) {}".format(i+1, s))
