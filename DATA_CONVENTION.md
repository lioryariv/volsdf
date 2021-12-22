# Data Convention

### Camera information and normalization
Besides multi-view RGB images, VolSDF needs cameras information in order to run. For each scan that we used, we supply a file named `cameras.npz`.
The `cameras.npz` file contains for each image its assosiacted camera projection matrix (named "world_mat_{i}"), and a normalization matrix (named "scale_mat_{i}").
#### Camera projection matrix
A 3x4 camera projection matrix, P = K[R | t] projects points from 3D coordinates to image pixels by the formula: d[x; y; 1]=P[X;Y;Z;1] where K is a 3x3 calibration matrix, [R t] is 3x4 a world to camera Euclidean transformation, [X;Y;Z] is the 3D point, [x;y] is the 2D pixel coordinates of the projected point and d is the depth of the point.
The input `cameras.npz` file contains the camera matrices, where P_i = cameras['world_mat_{i}'][:3, :] is a 3x4 matrix that projects points from the 3D world coordinates to the 2D coordinates of image i (intrinsics and extrinsics, i.e. P=K[R | t] ).
Each "world_mat" matrix is a concatenation of the camera projection matrix with a row vector of [0,0,0,1] (which makes it a 4x4 matrix).

#### Normalization matrix
The `cameras.npz` contains also one normalization matrix named "scale_mat_{i}" (identical for all i) for changing the coordinates system such that the cameras and the region of interest are located inside a sphere with radius 3 located at the origin (more details are in the paper).


### Preprocess
For converting BlendedMVS cameras format to ours, run:
```
cd data/preprocess/
python parse_cameras_blendedmvs.py --blendedMVS_path [BLENDED_MVS_PATH] --output_cameras_file [OUTPUT_CAMERAS_NPZ_FILE] --scan_ind [BLENDED_MVS_SCAN_ID]
```

In order to generate a normalization matrix for each scan, we used the input camera projection matrices. A script that demonstrates this process is presented in: `data/preprocess/normalize_cameras.py`.
Note: in order to run the supplied scans, it is not required to run this script. 
For normalizing a given `cameras.npz` file run:
```
cd data/preprocess/
python normalize_cameras.py --input_cameras_file [INPUT_CAMERAS_NPZ_FILE] --output_cameras_file [OUTPUT_NORMALIZED_CAMERAS_NPZ_FILE] [--number_of_cams [NUMBER_OF_CAMERAS_LIMIT]]
```
where the last argument is optional and used for limiting the number of cameras such that only the first [NUMBER_OF_CAMERAS_LIMIT] cameras are considered, which is useful for the DTU dataset, where for scan_id<80 only the first 49 cameras out of 64 are used.   

