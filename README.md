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


2. Cleaning

```
ETHD3D/build/PointCloudCleaner \
--in /path/to/cloud_lidar.ply \
--filter <5,10>
```
(local outliers removal, doesn't remove isolated points)
or
```
pcl_util/build/CloudSOR \
--input/path/to/cloud_lidar.ply \
--output /path/to/cloud_lidar_filtered.ply \
--knn 5 --std 6
```

3. Video frame addition to COLMAP db file

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
--img_dir /path/to/pictures/videos \
--colmap_img_root /path/to/images \
--maskroot /path/to/images_mask \
--batch_size 8
```

```
colmap feature_extractor \
--database_path /path/to/scan.db \
--image_path /path/to/images \
--image_list_path /path/to/vimages/video_frames_for_thorough_scan.txt
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
```

7. Third COLMAP step : thorough mapping.

```
mkdir -p output/sparse
colmap mapper --Mapper.multiple_models 0 --database_path scan.db --output_path /path/to/thorough/ --image_path images
```

This will create a model file in the folder `output/sparse` (or `output/sparse/0`), in the form of 3 files
```
└── sparse
    └── 0
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        └── project.ini
```

8. Fourth COLMAP step : [georeferencing](https://colmap.github.io/faq.html#geo-registration)

```
colmap model_aligner \
--input_path /path/to/thorough/0/ \
--output_path /path/to/geo-registered-model \
--ref_images_path /path/to/images/georef.txt
--robust_alignment_max_error 5
```
