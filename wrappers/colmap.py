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

    def map(self, output_model, input_model=None, multiple_models=False, start_frame_id=None):
        options = ["mapper", "--database_path", self.db,
                   "--image_path", self.image_path,
                   "--output_path", output_model,
                   "--Mapper.tri_ignore_two_view_tracks", "0"]
        if start_frame_id is not None:
            options += ["--Mapper.init_image_id1", str(start_frame_id)]
        if not multiple_models:
            options += ["--Mapper.multiple_models", "0"]
        if input_model is not None:
            options += ["--input_path", input_model]
            options += ["--Mapper.fix_existing_images", "1"]
        self.__call__(options)

    def register_images(self, output_model, input_model):
        options = ["image_registrator", "--database_path", self.db,
                   "--output_path", output_model,
                   "--input_path", input_model]
        self.__call__(options)

    def adjust_bundle(self, output_model, input_model, num_iter=10):
        options = ["bundle_adjuster",
                   "--output_path", output_model,
                   "--input_path", input_model,
                   "--BundleAdjustment.refine_extra_params", "0",
                   "--BundleAdjustment.max_num_iterations", str(num_iter)]
        self.__call__(options)

    def triangulate_points(self, output_model, input_model):
        options = ["point_triangulator",
                   "--database_path", self.db,
                   "--image_path", self.image_path,
                   "--output_path", output_model,
                   "--input_path", input_model]
        self.__call__(options)

    def align_model(self, output_model, input_model, ref_images, max_error=5):
        options = ["model_aligner", "--input_path", input_model,
                   "--output_path", output_model, "--ref_images_path",
                   ref_images, "--robust_alignment_max_error", str(max_error)]
        self.__call__(options)

    def export_model(self, output_path, input_model, output_type="PLY"):
        options = ["model_converter", "--input_path", input_model,
                   "--output_path", output_path, "--output_type", output_type]
        self.__call__(options)
