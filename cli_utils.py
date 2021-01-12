from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np


def add_main_options(parser):
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
    main_parser.add_argument('--add_new_videos', action="store_true",
                             help="If selected, will skip first 6 steps to directly register videos without mapping")
    main_parser.add_argument('--generate_groundtruth_for_individual_images', '--gt_images', action="store_true",
                             help="If selected, will generate Ground truth for individual images as well as videos")
    main_parser.add_argument('--save_space', action="store_true")
    main_parser.add_argument('-v', '--verbose', action="count", default=0)
    main_parser.add_argument('--resume_work', action="store_true",
                             help='If selected, will try to skip video aready localized, and ground truth already generated')
    main_parser.add_argument('--inspect_dataset', action="store_true",
                             help='If selected, will open a window to inspect the dataset. '
                                  'See https://github.com/ETH3D/dataset-pipeline#dataset-inspection')
    main_parser.add_argument('--vid_ext', nargs='+', default=[".mp4", ".MP4"],
                             help='Video extensions to scrape from input folder')
    main_parser.add_argument('--pic_ext', nargs='+', default=[".jpg", ".JPG", ".png", ".PNG"],
                             help='Image extensions to scrape from input folder')
    main_parser.add_argument('--raw_ext', nargs='+', default=[".ARW", ".NEF", ".DNG"],
                             help='Raw Image extensions to scrape from input folder')


def add_pcp_options(parser):
    pcp_parser = parser.add_argument_group("PointCloud preparation")
    pcp_parser.add_argument("--pointcloud_resolution", default=None, type=float,
                            help='If set, will subsample the Lidar point clouds at the chosen resolution')
    pcp_parser.add_argument("--SOR", default=[10, 6], nargs=2, type=float,
                            help="Satistical Outlier Removal parameters : Number of nearest neighbours, max relative distance to standard deviation")
    pcp_parser.add_argument('--registration_method', choices=["simple", "eth3d", "interactive"], default="simple",
                            help='Method used for point cloud registration. See README, Manual step by step : step 11')


def add_ve_options(parser):
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
    ve_parser.add_argument('--generic_model', default='OPENCV',
                           help='COLMAP model for generic videos. Same zoom level assumed throughout the whole video. '
                           'See https://colmap.github.io/cameras.html')
    ve_parser.add_argument('--full_metadata', default=None,
                           help='where to save all concatenated metadata in a file that will be used to add new videos afterward. '
                                'If not set, will save at the root of workspace')


def add_exec_options(parser):
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


def add_pm_options(parser):
    pm_parser = parser.add_argument_group("Photogrammetry")
    pm_parser.add_argument('--max_num_matches', default=32768, type=int, help="max number of matches, lower it if you get GPU memory error")
    pm_parser.add_argument('--match_method', default='vocab_tree', choices=['vocab_tree', 'exhaustive'],
                           help='Match method for first thorough photogrammetry, '
                                'see https://colmap.github.io/tutorial.html#feature-matching-and-geometric-verification')
    pm_parser.add_argument('--vocab_tree', type=Path, default="vocab_tree_flickr100K_words256K.bin")
    pm_parser.add_argument('--triangulate', action="store_true")
    pm_parser.add_argument('--multiple_models', action='store_true', help='If selected, will let colmap mapper do multiple models.'
                                                                          'The biggest one will then be chosen')
    pm_parser.add_argument('--more_sift_features', action="store_true",
                           help="If selected, will activate the COLMAP options ` SiftExtraction.domain_size_pooling` "
                                " and `--SiftExtraction.estimate_affine_shape` during feature extraction. Be careful, "
                                "this does not use GPU and is thus very slow. More info : "
                                "https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points")
    pm_parser.add_argument('--filter_models', action="store_true",
                           help="If selected, will filter video localization to smooth trajectory")
    pm_parser.add_argument('--stereo_min_depth', type=float, default=0.1, help="Min depth for PatchMatch Stereo")
    pm_parser.add_argument('--stereo_max_depth', type=float, default=100, help="Max depth for PatchMatch Stereo")


def add_om_options(parser):
    om_parser = parser.add_argument_group("Occlusion Mesh")
    om_parser.add_argument('--normals_method', default="radius", choices=["radius", "neighbours"],
                           help='Method used for normal computation between radius and nearest neighbours')
    om_parser.add_argument('--normals_radius', default=0.2, type=float,
                           help='If radius method for normals, radius within which other points will be considered neighbours')
    om_parser.add_argument('--normals_neighbours', default=8, type=int,
                           help='If nearest neighbours method chosen, number of neighbours to consider.'
                                'Could be very close or very far points, but has a constant complexity')
    om_parser.add_argument('--mesh_resolution', default=0.2, type=float,
                           help='Mesh resolution for occlusion in meters. Higher means more coarse. (in meters)')
    om_parser.add_argument('--splats', action='store_true',
                           help='If selected, will create splats for points in the cloud that are far from the occlusion mesh')
    om_parser.add_argument('--splat_threshold', default=0.1, type=float,
                           help='Distance from occlusion mesh at which a splat will be created for a particular point (in meters)')
    om_parser.add_argument('--max_splat_size', default=None, type=float,
                           help='Splat size is defined by mean istance from its neighbours. You can define a max splat size for '
                                'isolated points which otherwise would make a very large useless splat. '
                                'If not set, will be `2.5*splat_threshold`.')


def add_gt_options(parser):
    gt_parser = parser.add_argument_group("Ground Truth Creator")
    gt_parser.add_argument('--max_occlusion_depth', default=250, type=float,
                           help='max depth for occlusion. Everything further will not be considered at infinity')
    gt_parser.add_argument('--eth3d_splat_radius', default=0.01, type=float,
                           help='see splat radius for ETH3D')
    im_size = gt_parser.add_mutually_exclusive_group()
    im_size.add_argument('--output_rescale', type=float, default=1,
                         help='Rescale images for depth ground truth')
    im_size.add_argument('--output_width', type=int, default=None,
                         help='width of output images and depth maps')


def set_full_argparser():
    parser = ArgumentParser(description='Main pipeline, from LIDAR pictures and videos to GT depth enabled videos',
                            formatter_class=ArgumentDefaultsHelpFormatter)

    add_main_options(parser)
    add_pcp_options(parser)
    add_ve_options(parser)
    add_exec_options(parser)
    add_pm_options(parser)
    add_om_options(parser)
    add_gt_options(parser)
    return parser


def set_full_argparser_no_lidar():
    parser = ArgumentParser(description='Main pipeline, from pictures and videos to GT depth enabled videos, '
                                        'using only COLMAP (No lidar)',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    add_main_options(parser)
    add_ve_options(parser)
    add_exec_options(parser)
    add_pm_options(parser)
    add_om_options(parser)
    add_gt_options(parser)

    nl_parser = parser.add_argument_group("Main pipeline no Lidar specific options")
    nl_parser.add_argument("--SOR", default=[10, 6], nargs=2, type=float,
                           help="Satistical Outlier Removal parameters : Number of nearest neighbours, "
                                "max relative distance to standard deviation. "
                                "This will be used for filtering dense reconstruction")
    return parser


def set_new_images_arparser():
    parser = ArgumentParser(description='Additional pipeline, to add new pictures to an already existing workspace, '
                                        'The main pipeline must be finished before launching this script, '
                                        ' at least until reconstruction alignment',
                            formatter_class=ArgumentDefaultsHelpFormatter)
    add_main_options(parser)
    add_exec_options(parser)
    add_pm_options(parser)
    add_om_options(parser)
    add_gt_options(parser)

    ni_parser = parser.add_argument_group("Add new pictures specific options")

    ni_parser.add_argument("--map_new_images", action="store_true",
                           help="if selected, will replace the 'omage_registrator' step with a full mapping step")
    ni_parser.add_argument("--bundle_adjuster_steps", default=100, type=int,
                           help="number of iteration for bundle adjustor after image registration")
    ni_parser.add_argument("--rebuild_occlusion_mesh", action="store_true",
                           help="If selected, will rebuild a new dense point cloud and deauney mesh. "
                                "Useful when new images see new parts of the model")
    ni_parser.add_argument('--generic_model', default='OPENCV',
                           help='COLMAP model for added pictures. Same zoom level assumed throughout whole folders. '
                           'See https://colmap.github.io/cameras.html')
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
                   "Create video with GT visualization and Convert to KITTI format"]


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


def get_matrix(path):
    if path.isfile():
        '''Note : We use the inverse matrix here, because in general, it's easier to register the reconstructed model into the lidar one,
        as the reconstructed will have less points, but in the end we need the matrix to apply to the lidar point to be aligned
        with the camera positions (ie the inverse)'''
        return np.linalg.inv(np.fromfile(path, sep=" ").reshape(4, 4))
    else:
        print("Error, no registration matrix can be found")
        print("Ensure that your registration matrix was saved under the name {}".format(path))
        decision = None
        while decision not in ["y", "n", ""]:
            decision = input("retry ? [Y/n]")
        if decision.lower() in ["y", ""]:
            return get_matrix(path)
        elif decision.lower() == "n":
            return np.eye(4)
