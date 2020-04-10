from .default_wrapper import Wrapper


class Colmap(Wrapper):
    """docstring for Colmap"""

    def __init__(self, db, image_path, mask_path, binary="colmap", logfile=None, quiet=False):
        super().__init__(binary, quiet, logfile)
        self.db = db
        self.image_path = image_path
        self.mask_path = mask_path

    def extract_features(self, per_sub_folder=False, image_list=None, model="RADIAL", fine=False):
        options = ["feature_extractor", "--database_path", self.db,
                   "--image_path", self.image_path, "--ImageReader.mask_path", self.mask_path,
                   "--ImageReader.camera_model", model]
        if per_sub_folder:
            options += ["--ImageReader.single_camera_per_folder", "1"]
        if image_list is not None:
            options += ["--image_list_path", image_list]
        if fine:
            options += ["--SiftExtraction.domain_size_pooling", "1",
                        "--SiftExtraction.estimate_affine_shape", "1"]
        self.__call__(options)

    def match(self, method="exhaustive", guided_matching=True, vocab_tree=None):
        options = ["{}_matcher".format(method),
                   "--database_path", self.db,
                   "--SiftMatching.guided_matching", "1" if guided_matching else "0"]
        if method == "sequential":
            assert vocab_tree is not None
            options += ["--SequentialMatching.loop_detection", "1",
                        "--SequentialMatching.vocab_tree_path", vocab_tree]

        self.__call__(options)

    def map(self, output, input=None, multiple_models=False, start_frame_id=None):
        options = ["mapper", "--database_path", self.db,
                   "--image_path", self.image_path,
                   "--output_path", output]
        if start_frame_id is not None:
            options += ["--Mapper.init_image_id1", str(start_frame_id)]
        if not multiple_models:
            options += ["--Mapper.multiple_models", "0"]
        if input is not None:
            options += ["--input_path", input]
            options += ["--Mapper.fix_existing_images", "1"]
        self.__call__(options)

    def register_images(self, output, input):
        options = ["image_registrator", "--database_path", self.db,
                   "--output_path", output,
                   "--input_path", input]
        self.__call__(options)

    def adjust_bundle(self, output, input, num_iter=10):
        options = ["bundle_adjuster",
                   "--output_path", output,
                   "--input_path", input,
                   "--BundleAdjustment.refine_extra_params", "0",
                   "--BundleAdjustment.max_num_iterations", str(num_iter)]
        self.__call__(options)

    def triangulate_points(self, output, input, clear_points=True):
        options = ["point_triangulator",
                   "--database_path", self.db,
                   "--image_path", self.image_path,
                   "--output_path", output,
                   "--input_path", input]
        if clear_points:
            options += ["--clear_points", "1"]
        self.__call__(options)

    def align_model(self, output, input, ref_images, max_error=5):
        options = ["model_aligner", "--input_path", input,
                   "--output_path", output, "--ref_images_path",
                   ref_images, "--robust_alignment_max_error", str(max_error)]
        self.__call__(options)

    def export_model(self, output, input, output_type="PLY"):
        options = ["model_converter", "--input_path", input,
                   "--output_path", output, "--output_type", output_type]
        self.__call__(options)

    def undistort(self, output, input, max_image_size=1000):
        options = ["image_undistorter", "--image_path", self.image_path,
                   "--input_path", input, "--output_path", output,
                   "--output_type", "COLMAP", "--max_image_size", str(max_image_size)]
        self.__call__(options)

    def dense_stereo(self, workspace):
        options = ["patch_match_stereo", "--workspace_path", workspace,
                   "--workspace_format", "COLMAP",
                   "--PatchMatchStereo.geom_consistency", "1"]
        self.__call__(options)

    def stereo_fusion(self, workspace, output_model):
        options = ["stereo_fusion" "--workspace_path", workspace,
                   "--workspace_format", "COLMAP",
                   "--input_type", "geometric",
                   "--output_path", output_model]
        self.__call__(options)

    def merge_models(self, output, input1, input2):
        options = ["model_merger", "--output_path", output,
                   "--input_path1", input1,
                   "--input_path2", input2]
        self.__call__(options)

    def index_images(self, vocab_tree_output, vocab_tree_input):
        options = ["vocab_tree_retriever", "--database_path", self.db,
                   "--vocab_tree_path", vocab_tree_input, "--output_index", vocab_tree_output]
        self.__call__(options)
