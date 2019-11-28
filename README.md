# Photogrammetry and georegistration tools for Parrot drone videos

This is a set of python scripts used to construct a depth validation set with a Lidar generated point cloud.

To be extended but roughly, here are the key steps of the dataset creation :

1. Data acquisition on a particular scene
    - Make a photogrammetry flight plan with any drone, You can use e.g. the Anafi with the Pix4D capture app. It is important that pictures have GPS iunfo in the exif
    - Make some natural flights in the same scene, use either a Bebop2 or a Anafi to be able to use the PDraw tool
    - Make a Lidar scan of this very scene, and clean the resulting 3D point cloud : this is a crucial part as Lidar data will be assumed perfect for the rest of the workflow. You need to also note the projection system used (e.g. `EPSG 2154`) for geo registration. The file will a priori be a `.las` file with float64 values.

 2. Convert the `.las` float64 point cloud into a `.ply` float32
     - As 3D values are global, x and y will be huge. You need to make the cloud 0-centered by subtracting its centroid to it.
     - The centroid needs to be logged somewhere for future frame registration
     - This step is done by the script `las2ply.py`

2. Generate sky maps of your drone pictures to help the photogrammetry filter out noise during matching
    - Use a Neural Network to segment the drone picture and generate masks so that the black areas will be ignored
    - This is done with the script `generate_sky_masks.py`

3. Perform a photogrammetry on your pictures
    - The recommended tool is Colmap because further tools will use its output format
    - Use your photogrammetry dedicated pictures (obtained with pix4D capture) with a small subset of frames from your piloting videos to get a sparse 3D model, exhaustive enough to : 
        - Be matched with the Lidar Point cloud
        - Localize other video frames in the reconstruction
    - This picture set should be around 1000 frames to prevent the reconstruction from lasting too long.
    - The photogrammetry picture should already have GPS data in the exif to help colmap, but you also need to add them to the extracted frames from videos. This is done with the script `extract_video_with_gps.py`

4. Change reconstructed point cloud with shift and scale to match Lidar point cloud
    - Get the GPS coordinates of each picture from EXIF data, and deduce the *XYZ* coordinates in the projection system used by the Lidar point cloud
    - Substract to these coordinates the centroid that logged when converting the LAS file to PLY.
    - Log image filename and centered *XYZ* position in a file for georegistration of the reconstruction point cloud
    - This is done with the script `align_frames_with_lidar.py`
    - See [here]() for point cloud georegistration with colmap
    - Save the reconstructed point cloud to a ply file

5. Match the Lidar Point cloud and Reconstruction point cloud together with ICP.
    - The georegistration of the reconstructed point cloud should be sufficient to get a good starting point.
    - Done with PCL (to extend)

6. For each video : register all the video frames in the reconstructed model
    - It is not recommended to do it with all the videos at the same time, or the last bundle_adjustement will be very slow.
    - See [here]() for more details on frames registration


## Detailed method with the "Manoir" example

### Scene presentation

### Data acquisition


### Set-by-step commands

1. Extract frames from the videos. For the moment, extract only frames that will help the general reconstruction.

2. Put images in the following tree strcture:
```
├── images
│   ├── anafi
│   │   ├── raw
│   │   ├── rectilinear
│   │   └── video
│   │       ├── 4K30
│   │       └── 720p120
│   └── apn
└── images_mask
```

3. Generate masks in the folder `images_mask`

`python generate_sky_map.py --root images`

5. First COLMAP step : feature extraction

```
colmap feature_extractor --database scan.db --image
```

6. Second COLMAP step : matching. For less than 1000 images, you can use exhaustive matching (this will take around 2hours). If there is too much images, you can use either spatial matching or vocab tree matching

```
colmap exhaustive_matcher --database scan.db
```
or
```
colmap spatial_matcher --database scan.db
```
or
```
colmap vocab_tree_matcher --database scan.db
```

7. Third COLMAP step : mapping.

```
mkdir -p output/sparse
colmap mapper --Mapper.multiple_models 0 --database_path scan.db --output_path output/sparse/ --image_path images
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

8. Third COLMAP step : [georeferencing](https://colmap.github.io/faq.html#geo-registration)
    * Create an `images.txt` with the image name and their XYZ position, relative to the centered Lidar Point using the file `cloud_centroid.txt`created at step 4
    ```
    python align_frames_with_lidar.py --root images --centroid_path cloud_centroid.txt --output images.txt
    ```
    This will create a file with image paths and corresponding XYZ coordinates at each line.
    * ```
mkdir -p output/georeferenced
colmap model_aligner --input_path output/sparse --output_sparse output/georeferenced --ref_images_path images.txt
