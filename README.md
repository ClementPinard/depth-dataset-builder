# Photogrammetry and georegistration tools for Parrot drone videos

This is a set of python scripts  and c++ programs used to construct a depth validation set with a Lidar generated point cloud.
For a brief recap of what it does, see section [How it works](#how-it-works)

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

## Hardaware dependecies

To recreate the results of the study, you will need these hardware pieces :
 - Parrot Anafi
 - DJI Matrice 600
 - Velodyne Puck VLP16

Note that for our study, we provided the Anafi drone (\~700€), and the point cloud was created by a private company (\~3500€ for the whole scan process)


# How it works

Here are the key steps of the dataset creation :

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
    - This step is done by the script `video_to_colmap.py`

4. Georeference your images.
    - For each frame with *GPS* position, convert them in *XYZ* coorindates in the projection system used by the Lidar point cloud (Here, EPSG:2154 was used)
    - Substract to these coordinates the centroid that logged when converting the LAS file to PLY.
    - Log image filename and centered *XYZ* position in a file for georegistration of the reconstruction point cloud
    - This step is also done by the script `video_to_colmap.py`

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


## Detailed method with the "Manoir" example

### Scene presentation

### Data acquisition


### Running the full script


Structure your input folder so that it looks like this:
```
├── Pictures
│   ├── anafi
│   │   ├── raw
│   │   ├── rectilinear
│   └── apn
├── Videos
│   ├── 4K30
│   └── 720p120
└── Lidar
```

You can run the whole script with ```python main_pipeline.py```

#### Parameters brakdown

### Manual step by step

This will essentially do the same thing as the script, in order to let you change some steps at will.

1. Point cloud preparation

    ```
    python las2ply.py /path/to/cloud.las \
    --output_ply /path/to/cloud_lidar.ply
    --output_txt /path/to/centroid.txt
    ```


2. Point Cloud Cleaning
    For each ply file :

    ```
    ETHD3D/build/PointCloudCleaner \
    --in /path/to/cloud_lidar.ply \
    --filter <5,10>
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

    Optionally, if we have multiple lidar scans (which is not the case here), we can run a registration step with ETH3D

    ```
    ETHD3D/build/ICPScanAligner \
    -i /path/to/lidar.mlp \
    -o /path/to/lidar.mlp
    --number_of_scales 5
    ```

4. Video frame addition to COLMAP db file

    ```
    python video_to_colmap \
    --video_folder /path/to/videos \
    --system epsg:2154 \
    --centroid_path /path/to/centroid.txt \
    --output_folder /path/to/pictures/videos \
    --nw /path/to/anafi/native-wrapper.sh \
    --fps 1 \
    --total_frames 1000 \
    --save_space \
    --thorough_db /path/to/scan.db

    ```

    The video to colmap step will populate the scan db with new entries with the right camera parameters. And select a spatially optimal subset of frames from the full video for a photogrammetry with 1000 pictures.
    It will also create several txt files with list of file paths :

     - `video_frames_for_thorough_scan.txt` : all images used in the first thorough photogrammetry
     - `georef.txt` : all images with GPS position, and XYZ equivalent, with system and minus centroid of Lidar file.

     And finally, it will divide long videos into chunks with corresponding list of filepath so that we don't deal with too large sequences (limit here is 4000 frames)


5. First COLMAP step : feature extraction


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
    --image_list_path /path/to/images/video_frames_for_thorough_scan.txt
    --ImageReader.mask_path Path/to/images_mask/ \
    --ImageReader.camera_model RADIAL \
    --ImageReader.single_camera_per_folder 1 \
    ```

    We also recommand you make your own vocab_tree with image indexes, this will make the next matching steps faster.

    ```
    colmap vocab_tree_retriever \
    --database_path /path/to/scan.db\
    --vocab_tree_path /path/to/vocab_tree \
    --output_index /path/to/indexed_vocab_tree
    ```

6. Second COLMAP step : matching. For less than 1000 images, you can use exhaustive matching (this will take around 2hours). If there is too much images, you can use either spatial matching or vocab tree matching

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

7. Third COLMAP step : thorough mapping.

    ```
    mkdir -p /path/to/thorough/
    colmap mapper --Mapper.multiple_models 0 --database_path scan.db --output_path /path/to/thorough/ --image_path images
    ```

    This will create a model file in the folder `output/sparse` (or `output/sparse/0`), in the form of 3 files
    ```
    └── thorough
        └── 0
            ├── cameras.bin
            ├── images.bin
            ├── points3D.bin
            └── project.ini
    ```

    You can also add a last bundle adjustment using Ceres, supposedly better than the multicore used in mapper (albeit slower)

    ```
    colmap bundle_adjuster \
    --input_path /path/to/thorough/0
    --output_path /path/to/thorough/0
    ```

8. Fourth COLMAP step : [georeferencing](https://colmap.github.io/faq.html#geo-registration)

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

9. Video Localization
    All these substep will populate the db file, which is then used for matching. So you need to make a copy for each video.

    1. Extract all the frames of the video to same directory the `video_to_colmap.py` script exported the frame subset of this video.

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

    7. Filter the image sequence to exclude frame with an absurd acceleration and interpolate them instead
        ```
        python filter_colmap_model.py \
        --input_images_colmap /path/to/final_model/images.txt \
        --output_images_colmap /path/to/final_model/images.txt \
        --metadata /path/to/images/video/dir/metadata.csv \
        --interpolated_frames_list /path/to/images/video/dir/interpolated_frames.txt
        ```
    At the end of these per-video-tasks, you should have a model at `/path/to/georef_full` with all photogrammetry images + localization of video frames at 1fps, and for each video a TXT file with positions with respect to the first geo-registered reconstruction.

10. Point cloud densification

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

11. Point cloud registration
    
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

    Importante note : This operation doesn't work for scale adjustments which can be a problem with very large models.

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

    - Crop the /path/georef_dense.ply cloud, otherwise the Octomap will be very inefficient, and the cloud usually has very far outliers
    - Apply noise filtering on cropped cloud
    - Apply fine registration, with final overlap of 50%, scale adjustment, and Enable farthest point removal
    - save inverse of resulting registration

    For the fine registration part, as said earlier, the aligned cloud is the reconstruction and the reference cloud is the lidar

    finally, apply the registration matrix to `/path/to/lidar/mlp` to get `/path/to/registered.mlp`

    ```
    python meshlab_xml_writer.py transform \
    --input_meshlab /path/to/lidar.mlp \
    --output_meshlab /path/to/registered.mlp \
    --transform /path/to/registration_matrix.txt
    ```

12. Occlusion Mesh generation

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
    ```

    The ideal distance threshold is what is considered close range of the occlusion mesh, and the distance from which a splat (little square surface) will be created.

13. Raw Groundtruth generation
    
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

14. Dataset conversion

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






