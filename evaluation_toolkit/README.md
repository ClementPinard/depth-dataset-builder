# Evaluation Toolkit

Set of tools to run a particular algorithm on a dataset constructed with the validation set constructor, and evaluate it, along with advanced statistics regarding depth value and pixel position in image with respect to flight path vector.

## Inference Example

Get the last frame and a previous frame such that the displacement magnitude is as close to 30cm as possible, with the condition of having a rotation of less that 1 radian. Each frame is preprocessed so that it is of shape `[C, H, W]` and with a range `[0, 1]` instead of `[0, 255]`.

```python
from evaluation_toolkit import inferenceFramework

engine = inferenceFramework(dataset_root, evaluation_list, lambda x: x.transpose(2, 0, 1).astype(np.float32)[None]/255)

for sample in tqdm(engine):
    latest_frame, latest_intrinsics, _ = sample.get_frame()
    previous_frame, previous_intrinsics, previous_pose = sample.get_previous_frame(displacement=0.3)
    estimated_depth_map = my_model(latest_frame, previous_frame, previous_pose)
    engine.finish_frame(estimated_depth_map)
mean_inference_time, output_depth_maps = engine.finalize(output_path='output.npz')
```

You can find an example usage of this Inference Framework for SfmLearner [here](https://github.com/ClementPinard/SfmLearner-Pytorch/tree/validation_set_constructor)

## Evaluation

The evaluation step is a simple script that takes into input the computed depth maps (here in the file `output.npz`). You can combine multiple computed depth maps to compare algorithms.

```
depth_evaluation --dataset_root /path/to/dataset/root --est_depth output1.npz output2.npz --algorithm_names name1 name2 --evaluation_list_path /path/to/evaluation_list.txt --flight_path_vector_list /path/to/fligt_path_vector_list.txt <--scale_invariant> <--mask_path /path/to/mask.npy> --output_figures /path/to/figures/folder
```

It will output typical metrics and plot advanced statistics regarding the dataset and the depth estimations.
Note that if you want to save the figures, you will need `xelatex` installed in your system. Otherwise, don't specify a parameter to `--output_figures` and it will use `plt.show`