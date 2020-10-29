# Photogrammetry and georegistration tools for Parrot drone videos

This is a set of python scripts  and c++ programs used to construct a depth validation set with a Lidar generated point cloud.
For a brief recap of what it does, see section [How it works](#how-it-works)

## Table of contents

* [Software Dependencies](#software-dependencies)
* [Hardware Dependencies](#hardware-dependencies)
* [How it works](#how-it-works)
* [Step by step guide](#usage)
* [Detailed method with the manoir example](#detailed-method-with-the-manoir-example)


## Software Dependencies

These are the used tools, make sure to install them before running the scripts.

 - [CUDA](https://developer.nvidia.com/cuda-downloads)
 - [OpenCV](https://opencv.org/)
 - [ETH3D Dataset-pipeline](https://github.com/ETH3D/dataset-pipeline)
 - [Pytorch](https://pytorch.org/)
 - [COLMAP](https://colmap.github.io/)
 - [PDrAW from AnafiSDK](https://developer.parrot.com/docs/pdraw/)

Apart from CUDA, which you need to install by yourself, you can use the help script `install_dependencies.sh` to install them on ubuntu 20.04.

For PDrAW, there should be a `native-wrapper.sh` file that you to keep a track of. It's usually in `groundsdk/out/pdraw-linux/staging/native-wrapper.sh`(see [here](https://developer.parrot.com/docs/pdraw/installation.html))

For COLMAP, you will need a vocab tree for feature matching. You can download them at https://demuc.de/colmap/#download . In our tests, we took the 256K version.

## Hardware dependecies

To recreate the results of the study, you will need these hardware pieces :
 - Parrot Anafi
 - DJI Matrice 600
 - Velodyne Puck VLP16

Note that for our study, we provided the Anafi drone (\~700€), and the point cloud was created by a private company (\~3500€ for the whole scan process)


# How it works

Here are the key steps of the dataset creation :
See [Detailed method with the manoir example](#detailed-method-with-the-manoir-example) for a concrete example with options used.

1. Data acquisition on a particular scene
    - Make a photogrammetry flight plan with any drone, You can use e.g. the Anafi with the Pix4D capture app (it's free). It is important that pictures have GPS info in the exif
    - Make some natural flights in the same scene, use either a Bebop2 or a Anafi to be able to use the PDraw tool. In theory this is possible to adapt the current scripts to any IMU-GPS-powered camera.
    - Make a Lidar scan of this very scene, and clean the resulting 3D point cloud : this is a crucial part as Lidar data will be assumed perfect for the rest of the workflow. You need to also note the projection system used (e.g. `EPSG 2154`) for geo registration. The file will a priori be a `.las` file with float64 values.

2. Convert the `.las` float64 point cloud into a `.ply` float32
    - As 3D values are global, x and y will be huge. You need to make the cloud 0-centered by subtracting its centroid to it.
    - The centroid needs to be logged somewhere for future frame registration
    - This step is done by the script `las2ply.py`

3. Extract optimal frames from video for a thorough photogrammetry that will use a mix of pix4D flight plan pictures and video still frames.
    - The total number of frame must not be too high to prevent the reconstruction from lasting too long on a single desktop (we recommand between 500 an 1000 images)
    - At the same time, extract if possible information on camera parameters to identify which video sequences share the same parameters (e.g 4K videos vs 720p videos, or different levels of zooming)
    - This step is done by the script `videos_to_colmap.py`

4. Georeference your images.
    - For each frame with *GPS* position, convert them in *XYZ* coorindates in the projection system used by the Lidar point cloud (Here, EPSG:2154 was used)
    - Substract to these coordinates the centroid that logged when converting the LAS file to PLY.
    - Log image filename and centered *XYZ* position in a file for georegistration of the reconstruction point cloud
    - This step is also done by the script `videos_to_colmap.py`

4. Generate sky maps of your drone pictures to help the photogrammetry filter out noise during matching
    - Use a Neural Network to segment the drone picture and generate masks so that the black areas will be ignored
    - This is done with the script `generate_sky_masks.py`

3. Perform a photogrammetry on your pictures
    - The recommended tool is COLMAP because further tools will use its output format
    - You should get a sparse 3D model, exhaustive enough to : 
        - Be matched with the Lidar Point cloud
        - Localize other video frames in the reconstruction

4. Change reconstructed point cloud with shift and scale to match Lidar point cloud
    - See [here](https://colmap.github.io/faq.html#geo-registration) for point cloud georegistration with colmap

5. For each video : continue the photogrammetry with video frames at a low fps (we took 1fps)
    - We do this in order to keep the whole mapping at a linear time
    - Merge all the resulting models into one full model with thorough photogrammetry frames and all the 1fps frames
    - Finish registering the remaning frames. For RAM reasons, every video is divided into chunks, so that a sequence registered is never more than 4000 frames.
    - Filter the final model at full framerate : remove points with absurd angular and translational accleration. Interpolate the resulting discarded points (but keep a track of them). This is done in the script `filter_colmap_model.py`

6. Densify the resulting point cloud with COLMAP (see [here](https://colmap.github.io/tutorial.html#dense-reconstruction))
    - Export a PLY file along with the VIS file with `colmap stereo_fusion`

7. Match the Lidar Point cloud and full reconstruction point cloud together with ICP.
    - The georegistration of the reconstructed point cloud should be sufficient to get a good starting point.
    - By experience, the best method here is to use CloudCompare, but You can use ETH3D or PCL to do it
    - The resulting transformation matrix should be stored in a TXT file, the same way cloudcompare proposes to do it

8. Construct a PLY+VIS file pair based on lidar scan
    - The PLY file is basically every lidar point
    - The VIS file stores frames from which point is visible : we reuse the PLY+VIS from step 6, and assume a lidar point has the same visibility as the closest point in the denified reconstructed point.

9. Run the delauney mesher tool from COLMAP to construct an occlusion mesh

10. Construct the Splats with ETH3D

11. Construct the ground truth Depth with ETH3D

12. Visualize and Convert the resulting dataset to match the format of a more well known dataset, like KITTI.


## Usage

### Running the full script


Structure your input folder so that it looks like this:
```
├── Pictures
│   ├── anafi
│   │   ├── raw
│   │   └── rectilinear
│   └── apn
├── Videos
│   ├── 4K30
│   └── 720p120
└── Lidar
```

You can run the whole script with ```python main_pipeline.py```. If you don't have a lidar point cloud and want to use COLMAP reconstructed cloud as Groundtruth, you can use ```python main_pipeline_no_lidar.py``` which will be very similar, minus point cloud cleaning and registration steps.

#### Parameters breakdown

All the parameters for `main_pipeline.py` are defined in the file `cli_utils.ply`.You will find below a summary :

1. Main options
    * `--input_folder` : Input Folder with LAS/PLY point clouds, videos, and images, defined above
    * `--workspace` : Path to workspace where COLMAP operations will be done. It needs to be on a SSD, and size needed depends on video size, but should at least be 20 Go.
    * `--raw_output_folder` : Path to output folder for raw depth maps. Must be very big, especially with 4K videos. for 4K30fps video, count around 60Go per minute of video.
    * `--converted_output_folder` : Path to output folder for converted depth maps and visualization. Must be big but usually smaller than raw output because depth map is still uncompressed, but downscaled.
    * `--show_steps` : If selected, will make a dry run just to list steps and their numbers.
    * `--skip_step` : Skip the selected steps. Can be useful an operation is done manually)
    * `--begin_step` : Skip all steps before this step. Useful when the script failed at some point
    * `--resume_work` : If selected, will try to skip video aready localized, and ground truth already generated
    * `--inspect_dataset` : If selected, will open a window to inspect the dataset at key steps. See https://github.com/ETH3D/dataset-pipeline#dataset-inspection
    * `--save_space` : If selected, will try to save space in workspace by only extracting needed frames and removing them as soon as they are no longer needed. Strongly advised.
    * `--vid_ext` : Video extensions to scrape from input folder. By default will search for `mp4` and `MP4` files
    * `--pic_ext` : Same as Video extensions, but for Image. By default will search for `jpg`, `JPG`, `png`and `PNG` files.
    * `--raw_ext` : Same as Video extensions, but for RAW image. By default will search for `ARW`, `NEF` and  `DNG` files.

2. Executable files
    * `--nw` : Native wrapper location. See https://developer.parrot.com/docs/pdraw/installation.html#run-pdraw
    * `--colmap` : Colmap exec location. Usually just `Colmap` if it has been installed system-wide.
    * `--ffmpeg` : ffmpeg exec location. Usually just `ffmpeg` if it has been installed system-wide.
    * `--eth3d` : ETH3D dataset pipeline exec files folder location. Usually at `dataset-pipeline/build/`.
    * `--pcl_util` : PCL util exec files. Usually at `pcl_util/build` (source in this repo)
    * `--log` : If set, will output stdout and stderr of these exec files to a log file, which can be read from anther terminal with `tail`.

3. Lidar point cloud preparation

    * `--pointcloud_resolution` : If set, will subsample the Lidar point clouds at the chosen resolution.
    * `--SOR` : Satistical Outlier Removal parameters. This accepts 2 arguments : Number of nearest neighbours and max relative distance to standard deviation. See https://pcl.readthedocs.io/projects/tutorials/en/latest/statistical_outlier.html
    * `--registration_method` : Method use for point cloud registration, chose between "simple", "eth3d" and "interactive" ("simple" by default). See Manual step by step : step 11')

4. Video extractor

    * `--total_frames` : Total number of frames that will be used for the first thorough photogrammetry. By default 500, keep this number below 1000.
    * `--orientation_weight` : Weight applied to orientation during optimal sample. Higher means two pictures with same location but different orientation will be considered further apart.
    * `--resolution_weight` : Same as orientation, but with image size.
    * `--max_sequence_length` : COLMAP needs to load ALL the feature matches to register new frames. As such, some videos are too long to fit in RAM, and we need to divide the video in Chunks that will treated separately and then merged together. This parameter is the number max of frames for a chunk. Ideal value is around 500 frames for 1Go of RAM, regardless of resolution.
    * `--num_neighbours` : number of frames overlapping between chunks. This is for merge purpose.
    * `--system` : coordinates system used for GPS, should be the same as the LAS files used.
    * `--lowfps`: framerate at which videos will be scanned WITH reconstruction. 1fps by default
    * `--include_lowfps_thorough` : if selected, will include videos frames at lowfps for thorough scan (longer). This can be useful when some videos are not GPS localized (e.g. handhel camera) and are still relevant for the thorough photogrammetry.

5. Photogrammetry

    * `--max_num_matches` : Max number of matches, lower it if you get GPU memory error.
    * `--vocab_tree` : Pah to vocab tree, can be downloaded [here](https://demuc.de/colmap/#download)
    * `--multiple_models` : If selected, will let colmap mapper do multiple models. The biggest one will then be chosen
    * `--more_sift_features` : If selected, will activate the COLMAP options ` SiftExtraction.domain_size_pooling` and `--SiftExtraction.estimate_affine_shape` during feature extraction. Be careful, this does not use GPU and is thus very slow. More info : https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points
    * `--add_new_videos` : If selected, will skip the mapping steps to directly register new video with respect to an already existing colmap model.
    * `--stereo_min_depth` : Min depth for PatchMatch Stereo used during point cloud densification
    * `--stereo_max_depth` : Same as min depth but for max depth.

6. Occlusion Mesh
    * `--normals_method` : Method used for normal computation between radius and nearest neighbours.
    * `--normals_radius` : If radius method for normals, radius within which other points will be considered neighbours
    * `--normals_neighbours` : If nearest neighbours method chosen, number of neighbours to consider. Could be very close or very far points, but has a constant complexity.
    * `--mesh_resolution` : Mesh resolution for occlusion in meters. Higher means more coarse. (default 0.2, i.e. 20cm)
    * `--splats` : If selected, will create splats for points in the cloud that are far from the occlusion mesh.
    * `--splat_threshold` : Distance from occlusion mesh at which a splat will be created for a particular point (default, 10cm)
    * `--max_splate_size` : Splat size is defined by mean istance from its neighbours. You can define a max splat size for isolated points which otherwise would make a very large useless splat. If not set, will be `2.5*splat_threshold`.

### Manual step by step

This will essentially do the same thing as the script, in order to let you change some steps at will.

1. Point cloud preparation

    ```
    python las2ply.py /path/to/cloud.las \
    --output_folder /path/to/output_folder
    ```

    This will save a ply file along with a centroid file
     - `/path/to/output_folder/cloud.ply`
     - `/path/to/output_folder/centroid.txt`


2. Point Cloud Cleaning
    For each ply file :

    ```
    ETHD3D/build/PointCloudCleaner \
    --in /path/to/output_folder/cloud.ply \
    --filter 5,10
    ```
    (local outliers removal, doesn't necessarily remove isolated points)
    or
    ```
    pcl_util/build/CloudSOR \
    --input/path/to/cloud_lidar.ply \
    --output /path/to/cloud_lidar_filtered.ply \
    --knn 5 --std 6
    ```

3. Meshlab Project creation
    ```
    python meshlab_xml_writer.py create \
    --input_models /path/to/cloud1 [../path/to/cloudN] \
    --output_meshlab /path/to/lidar.mlp
    ```

    Optionally, if we have multiple lidar scans, we can run a registration step with ETH3D

    ```
    ETHD3D/build/ICPScanAligner \
    -i /path/to/lidar.mlp \
    -o /path/to/lidar.mlp
    --number_of_scales 5
    ```

4. First COLMAP step (divided in two parts) : feature extraction for photogrammetry frames

    ```
    python generate_sky_masks.py \
    --img_dir /path/to/images \
    --colmap_img_root /path/to/images \
    --maskroot /path/to/images_mask \
    --batch_size 8
    ```

    ```
    colmap feature_extractor \
    --database_path /path/to/scan.db \
    --image_path /path/to/images \
    --ImageReader.mask_path Path/to/images_mask/ \
    --ImageReader.camera_model RADIAL \
    --ImageReader.single_camera_per_folder 1 \
    ```

    We don't explicitely need to extract features before having video frames, but this will populate the `/path/to/scan.db` file with the photogrammetry pictures and corresponding id that will be reserved for future version of the file. Besides, it automatically set a camera per folder too.

5. Video frame addition to COLMAP db file

    ```
    python video_to_colmap.py \
    --video_folder /path/to/videos \
    --system epsg:2154 \
    --centroid_path /path/to/centroid.txt \
    --colmap_img_root /path/to/images \
    --nw /path/to/anafi/native-wrapper.sh \
    --fps 1 \
    --total_frames 1000 \
    --save_space \
    --thorough_db /path/to/scan.db
    ```

    The video to colmap step will populate the scan db with new entries with the right camera parameters, and select a spatially optimal subset of frames from the full video for a photogrammetry with 1000 pictures.
    It will also create several txt files with list of file paths :
     - `video_frames_for_thorough_scan.txt` : all images used in the first thorough photogrammetry
     - `georef.txt` : all images with GPS position, and XYZ equivalent, with system and minus centroid of Lidar file.

    The system parameter (here epsg:2154) is the one used in the point cloud. The geo localized frame will then be localized inside the point cloud, which will help register the COLMAP reconstructed point with the Lidar PointCloud. See more info [here](https://en.wikipedia.org/wiki/Spatial_reference_system). It must be compatible with [Proj](https://proj.org).

    And finally, it will divide long videos into chunks with corresponding list of filepath so that we don't deal with too large sequences (limit here is 4000 frames)


6. Second part of first COLMAP step : feature extraction for video frames used for thorough photogrammetry


    ```
    python generate_sky_masks.py \
    --img_dir /path/to/images \
    --colmap_img_root /path/to/images \
    --maskroot /path/to/images_mask \
    --batch_size 8
    ```

    (this is the same command as step 4)

    ```
    colmap feature_extractor \
    --database_path /path/to/scan.db \
    --image_path /path/to/images \
    --image_list_path /path/to/images/video_frames_for_thorough_scan.txt
    --ImageReader.mask_path Path/to/images_mask/ \
    ```

    We also recommand you make your own vocab_tree with image indexes, this will make the next matching steps faster. You can download a vocab_tree at https://demuc.de/colmap/#download : We took the 256K version in our tests.

    ```
    colmap vocab_tree_retriever \
    --database_path /path/to/scan.db\
    --vocab_tree_path /path/to/vocab_tree \
    --output_index /path/to/indexed_vocab_tree
    ```

7. Second COLMAP step : matching. For less than 1000 images, you can use exhaustive matching (this will take around 2hours). If there is too much images, you can use either spatial matching or vocab tree matching

    ```
    colmap exhaustive_matcher \
    --database_path scan.db \
    --SiftMatching.guided_matching 1
    ```
    or
    ```
    colmap spatial_matcher \
    --database scan.db \
    --SiftMatching.guided_matching 1
    ```
    or
    ```
    colmap vocab_tree_matcher \
    --database scan.db \
    --VocabTreeMatching.vocab_tree_path /path/to/indexed_vocab_tree
    --SiftMatching.guided_matching 1
    ```

8. Third COLMAP step : thorough mapping.

    ```
    mkdir -p /path/to/thorough/
    colmap mapper --database_path scan.db --output_path /path/to/thorough/ --image_path images
    ```

    This will create multiple models located in folder named `output/sparse/N` , `N`being a number, starting from 0. Each model will be, in the form of 3 files
    ```
    └── thorough
        └── N
            ├── cameras.bin
            ├── images.bin
            ├── points3D.bin
            └── project.ini
    ```

    COLMAP creates multiple models in the case the model has multiple sets of images that don't overlap. Most of the time, there will be only 1 model (named `0`). Depending on the frame used for initialization, it can happen that the biggest model is not the first. Here we will assume that it is indeed the first (`0`), but you are exepcted to change that number if it is not the most complete model COLMAP could construct.

    You can finally add a last bundle adjustment using Ceres, supposedly better than the multicore used in mapper (albeit slower)

    ```
    colmap bundle_adjuster \
    --input_path /path/to/thorough/0
    --output_path /path/to/thorough/0
    ```

9. Fourth COLMAP step : [georeferencing](https://colmap.github.io/faq.html#geo-registration)

    ```
    mkdir -p /path/to/geo_registered_model
    colmap model_aligner \
    --input_path /path/to/thorough/0/ \
    --output_path /path/to/geo_registered_model \
    --ref_images_path /path/to/images/georef.txt
    --robust_alignment_max_error 5
    ```

    This model will be the reference model, every further models and frames localization will be done with respect to this one.
    Even if we could, we don't run Point cloud registration right now, as the next steps will help us to have a more complete point cloud.

10. Video Localization
    All these substep will populate the db file, which is then used for matching. So you need to make a copy for each video.

    1. Extract all the frames of the video to same directory the `videos_to_colmap.py` script exported the frame subset of this video.

        ```
        ffmpeg \
        -i /path/to/video.mp4 \
        -vsync 0 -qscale:v 2 \
        /path/to/images/videos/dir/
        ```

    2. continue mapping with low fps images, use sequential matcher

        ```
        python generate_sky_masks.py \
        --img_dir /path/to/images/videos/dir \
        --colmap_img_root /path/to/images \
        --maskroot /path/to/images_mask \
        --batch_size 8
        ```

        ```
        python add_video_to_db.py \
        --frame_list /path/to/images/videos/dir/lowfps.txt \
        --metadata /path/to/images/videos/dir/metadata.csv\
        --database /path/to/video_scan.db
        ```

        ```
        colmap feature_extractor \
        --database_path /path/to/video_scan.db \
        --image_path /path/to/images \
        --image_list_path /path/to/images/videos/dir/lowfps.txt
        --ImageReader.mask_path Path/to/images_mask/
        ```

        ```
        colmap sequential_matcher \
        --database_path /path/to/video_scan.db \
        --SequentialMatching.loop_detection 1 \
        --SequentialMatching.vocab_tree_path /path/to/indexed_vocab_tree
        ```

        ```
        colmap mapper \
        --input /path/to/geo_registered_model \
        --output /path/to/lowfps_model \
        --Mapper.fix_existing_images 1
        ```

    3.  Re-georeference the model

        This is a tricky part : to ease convergence, the mapper normalizes the model, losing the initial georeferencing.
        To avoid this problem, we merge the model back to the first one. the order between input1 and input2 is important!

        ```
        colmap model_merger \
        --input1 /path/to/geo_registered_model \
        --input2 /path/to/lowfps_model \
        --output /path/to/lowfps_model
        ```

    4. Add mapped frame to the full model that will be used for Lidar registration

        ```
        colmap model_merger \
        --input1 /path/to/geo_registered_model \
        --input2 /path/to/lowfps_model \
        --output /path/to/georef_full
        ```

        For next videos, replace input1 with `/path/to/georef_full` , which will incrementally add more and more images to the model.

    5. Register the remaining frames of the videos, without mapping. This is done by chunks in order to avoid RAM problems.
        For each Chunk `n`, copy a copy of the scan database and do the same operations as above, minus the mapping, replaced with image registration.

        ```
        cp /path/to/video_scan.db /path/to/video_scan_chunk_n.db
        ```

        ```
        python add_video_to_db.py \
        --frame_list /path/to/images/videos/dir/full_n.txt \
        --metadata /path/to/images/videos/dir/metadata.csv\
        --database /path/to/video_scan_chunk_n.db
        ```

        ```
        colmap feature_extractor \
        --database_path /path/to/video_scan_chunk_n.db \
        --image_path /path/to/images \
        --image_list_path /path/to/images/videos/dir/full_n.txt
        --ImageReader.mask_path Path/to/images_mask/
        ```

        ```
        colmap sequential_matcher \
        --database_path /path/to/video_scan_chunk_n.db \
        --SequentialMatching.loop_detection 1 \
        --SequentialMatching.vocab_tree_path /path/to/indexed_vocab_tree
        ```

        ```
        colmap image_registrator \
        --database_path /path/to/video_scan_chunk_n.db \
        --input_path /path/to/lowfps_model
        --output_path /path/to/chunk_n_model
        ```

        (optional bundle adjustment)

        ```
        colmap bundle_adjuster \
        --input_path /path/to/chunk_n_model \
        --output_path /path/to/chunk_n_model \
        --BundleAdjustment.max_num_iterations 10
        ```

        if first chunk, simply copy `/path/to/chunk_n_model` to `/path/to/full_video_model`.
        Otherwise:

        ```
        colmap model_merger \
        --input1 /path/to/full_video_model \
        --input2 /path/to/chunk_n_model \
        --output /path/to/full_video_model
        ```

        At the end of this step, you should have a model with all the (localizable) frames of the videos + the other frames that where used for the first thorough photogrammetry

    6. Extract the frame position from the resulting model

        ```
        python extract_video_from_model.py \
        --input_model /path/to/full_video_model \
        --output_model /path/to/final_model \
        --metadata_path /path/to/images/video/dir/metadata.csv
        --output_format txt
        ```

    7. Filter the image sequence to exclude frame with an absurd acceleration and interpolate them instead. We keep a track of interpolated frames, which will not be used for depth validation but can be used for depth estimation algorithms that need odometry of previous frames.
        ```
        python filter_colmap_model.py \
        --input_images_colmap /path/to/final_model/images.txt \
        --output_images_colmap /path/to/final_model/images.txt \
        --metadata /path/to/images/video/dir/metadata.csv \
        --interpolated_frames_list /path/to/images/video/dir/interpolated_frames.txt
        ```
    At the end of these per-video-tasks, you should have a model at `/path/to/georef_full` with all photogrammetry images + localization of video frames at 1fps, and for each video a TXT file with positions with respect to the first geo-registered reconstruction.

11. Point cloud densification

    ```
    colmap image_undistorter \
    --image_path /path/to/images \
    --input_path /path/to/georef_full \
    --output_path /path/to/dense \
    --output_type COLMAP \
    --max_image_size 1000
    ```

    `max_image_size` option is optional but recommended if you want to save space when dealing with 4K images

    ```
    colmap patch_match_stereo \
    --workspace_path /path/to/dense \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency 1
    ```

    ```
    colmap stereo_fusion \
    --workspace_path /path/to/dense \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path /path/to/georef_dense.ply
    ```

    This will also create a `/path/to/georef_dense.ply.vis` file which describes frames from which each point is visible.

12. Point cloud registration
    
    Convert meshlab project to PLY with normals :

    Determine the transformation to apply to `/path/to/lidar.mlp` to get to `/path/to/georef_dense.ply` so that we can have the pose of the cameras with respect to the lidar.

    Option 1 : construct a meshlab project similar to `/path/to/lidar.mlp` with `/path/to/georef_dense.ply` as first mesh and run ETH3D's registration tool 
    ```
    python meshlab_xml_writer.py add \
    --input_models /path/to/georef_dense.ply \
    --start_index 0 \
    --input_meshlab /path/to/lidar.mlp \
    --output_meshlab /path/to/registered.mlp
    ```
    ```
    ETHD3D/build/ICPScanAligner \
    -i /path/to/registered.mlp \
    -o /path/to/registered.mlp \
    --number_of_scales 5
    ```

    The second matrix in `/path/to/register.mlp` will be the matrix transform from `/path/to/lidar.mlp` to `/path/to/georef_dense.ply`

    Importante note : This operation doesn't work for scale adjustments. Theoretically, if the video frames are gps localized, it should no be a problem, but it can be a problem with very large models where a small scale error will be responsible for large displacement errors locally.

    Option 2 : construct a PLY file from lidar scans and register the reconstructed cloud with respect to the lidar, with PCL or CloudCompare. We do this way (and not from lidar to reconstructed), because it is usually easier to register the cloud with less points with classic ICP)
    ```
    ETHD3D/build/NormalEstimator \
    -i /path/to/lidar.mlp \
    -o /path/to/lidar_with_normals.ply
    ```

    ```
    pcl_util/build/CloudRegistrator \
    --georef /path/to/georef_dense.ply \
    --lidar /path/to/lidar_with_normals.ply \
    --output_matrix /path/toregistration_matrix.txt
    ```

    Note that `/path/toregistration_matrix.txt`stored the inverse of the matrix we want, so you have to invert it and save back the result.

    Or use CloudCompare : https://www.cloudcompare.org/doc/wiki/index.php?title=Alignment_and_Registration
    Best results were maintened with these consecutive steps :
    - Crop the /path/georef_dense.ply cloud, otherwise the Octomap will be very inefficient, and the cloud usually has very far outliers. See [Cross section](https://www.cloudcompare.org/doc/wiki/index.php?title=Cross_Section).
    - Apply noise filtering on cropped cloud . See [Noise filter](https://www.cloudcompare.org/doc/wiki/index.php?title=Noise_filter).
    - (Optional, especially if the frames are gps localized) Manually apply a rough registration with point pair picking. See [Align](https://www.cloudcompare.org/doc/wiki/index.php?title=Align).
    - Apply fine registration, with final overlap of 50%, scale adjustment, and Enable farthest point removal. See [ICP](https://www.cloudcompare.org/doc/wiki/index.php?title=ICP)
    - Save resulting registration matrix

    For the fine registration part, as said earlier, the aligned cloud is the reconstruction and the reference cloud is the lidar

    finally, apply the registration matrix to `/path/to/lidar/mlp` to get `/path/to/registered.mlp`

    ```
    python meshlab_xml_writer.py transform \
    --input_meshlab /path/to/lidar.mlp \
    --output_meshlab /path/to/registered.mlp \
    --transform /path/to/registration_matrix.txt
    --inverse
    ```

13. Occlusion Mesh generation

    Use COLMAP delaunay mesher to generate a mesh from PLY + VIS.
    Normally, COLMAP expect the cloud it generated when running the `stereo_fusion` step, but we use the lidar point cloud instead.

    Get a PLY file for the registered lidar point cloud
    
    ```
    ETHD3D/build/NormalEstimator \
    -i /path/to/registered.mlp \
    -o /path/to/lidar_with_normals.ply
    ```

    ```
    pcl_util/build/CreateVisFile \
    --georef_dense /path/to/georef_dense.ply \
    --lidar lidar_with_normals.ply \
    --output_cloud /path/to/dense/fused.ply \
    --resolution 0.2
    ```

    This is important to place the resulting point cloud at root of COLMAP MVS workspace `/path/to/dense` that was used for generating `/path/to/georef_dense.ply` and name it `fused.ply` because it is hardwritten on COLMAP's code.
    The file `/path/to/fused.ply.vis` will also be generated. The resolution option is used to reduce the computational load of the next step.

    ```
    colmap delaunay_mesher \
    --input_type dense \
    --input_path /path/to/dense \
    --output_path /path/to/occlusion_mesh.ply
    ```

    Generate splats for lidar points outside of occlusion mesh close range. See https://github.com/ETH3D/dataset-pipeline#splat-creation

    ```
    ETH3D/build/SplatCreator \
    --point_normal_cloud_path /path/tolidar_with_normals.ply \
    --mesh_path /path/to/occlusion_mesh.ply \
    --output_path /path/to/splats.ply
    --distance_threshold 0.1
    --max_splat_size 0.25
    ```

    The ideal distance threshold is what is considered close range of the occlusion mesh, and the distance from which a splat (little square surface) will be created.

14. Raw Groundtruth generation
    
    For each video :

    ```
    ETH3D/build/GroundTruthCreator \
    --scan_alignment_path /path/to/registered.mlp \
    --image_base_path /path/to/images \
    --state_path path/to/final_model \
    --output_folder_path /path/to/raw_GT \
    --occlusion_mesh_path /path/to/occlusion_mesh.ply \
    --occlusion_splats_path /path/to/splats/ply \
    --max_occlusion_depth 200 \
    --write_point_cloud 0 \
    --write_depth_maps 1 \
    --write_occlusion_depth 1 \
    --compress_depth_maps 1
    ```

    This will create for each video a folder `/path/to/raw_GT/groundtruth_depth/<video name>/` with compressed files with depth information. Option `--write_occlusion_depth` will make the folder `/path/to/raw_GT/` much heavier but is optional. It is used for inspection purpose.

15. Dataset conversion

    For each video :

    ```
    python convert_dataset.py \
    --depth_dir /path/to/raw_GT/groundtruth_depth/<video name>/ \
    --images_root_folder /path/to/images/ \
    --occ_dir /path/to/raw_GT/occlusion_depth/<video name>/ \
    --metadata_path /path/to/images/videos/dir/metadata.csv \
    --dataset_output_dir /path/to/dataset/ \
    --video_output_dir /path/to/vizualisation/ \
    --interpolated_frames_list /path/to/images/video/dir/interpolated_frames.txt \
    --final_model /path/to/final_model/ \
    --video \
    --downscale 4 \
    --threads 8
    ```

## Detailed method with the "Manoir" example

### Scene presentation

### Data acquisition