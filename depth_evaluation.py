from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = ArgumentParser(description='Convert EuroC dataset to COLMAP',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_root', metavar='DIR', type=Path)
parser.add_argument('--est_depth', metavar='DIR', type=Path,
                    help='where the depth maps are stored, must be a 3D npy file')
parser.add_argument('--evaluation_list', metavar='PATH', type=Path,
                    help='File with list of images to test for depth evaluation')
parser.add_argument('--flight_path_vector_list', metavar='PATH', type=Path,
                    help='File with list of speed vectors, used to compute error wrt direction')
parser.add_argument('--scale-invariant', action='store_true',
                    help='If selected, will rescale depth map with ratio of medians')

coord = None


def get_values(gt_depth, estim_depth, fpv):
    global coords
    if coords is None:
        coords = np.stack(np.meshgrid(np.arange(gt_depth.shape[0]), np.arange(gt_depth.shape[1])), axis=-1)
    fpv_dist = np.linalg.norm(coords - fpv, axis=-1)
    valid = gt_depth == gt_depth & gt_depth < np.inf
    fpv_dist = fpv_dist[valid]
    valid_coords = coords[valid[..., None]]
    values = np.stack([gt_depth[valid], estim_depth[valid], *valid_coords.T, fpv_dist], axis=-1)

    return pd.DataFrame(values, columns=["GT", "estim", "x", "y", "fpv_dist"])


def plot_distribution(bins, values, ax):
    bin_dists = bins[1:] - bins[:-1]
    total = sum(bin_dists)
    normalized_values = values * bin_dists / total
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    ax.plot(bin_centers, normalized_values)


def main():
    args = parser.parse_args()
    n_bins = 10
    with open(args.evaluation_list, 'r') as f:
        depth_paths = [line[:-1] for line in f.readlines()]
    fpv_list = np.readtxt(args.flight_path_vector_list)
    estimated_depth = np.load(args.est_depth)
    values_df = None
    assert(len(depth_paths) == estimated_depth.shape[0])
    for filepath, current_estimated_depth, fpv in zip(depth_paths, estimated_depth, fpv_list):

        GT = np.load(filepath)
        new_values = get_values(GT, current_estimated_depth, fpv)
        if values_df is None:
            values_df = new_values
        else:
            values_df = values_df.append(new_values)

    values_df["log_GT"] = np.log(values_df["GT"])
    values_df["log_estim"] = np.log(values_df["estim"])
    values_df["diff"] = np.abs(values_df["GT"] - values_df["estim"])
    values_df["reldiff"] = values_df["diff"] / values_df["GT"]
    values_df["logdiff"] = np.abs(values_df["log_GT"] - values_df["log_estim"])

    plot = True
    if plot:

        def error_map(series):
            error_per_px = series.groupby(by=["x", "y"]).mean()
            error_map = np.full(estimated_depth.shape[:2], np.NaN)
            error_map[error_per_px.index] = error_per_px.values
            return error_map

        min_gt = values_df["GT"].min()
        max_gt = values_df["GT"].max()

        bins = np.linspace(min_gt, max_gt, n_bins + 1)

        estim_per_GT = {}
        for b1, b2 in zip(bins[:-1], bins[1:]):
            per_gt = values_df[values_df["GT"] > b1 & values_df["GT"] < b2]
            estim_per_GT[(b1+b2)/2] = {"normal": np.histogram(per_gt["estim"]),
                                       "log_normal": np.histogram(per_gt["log_estim"])}

        global_diff = np.histogram(values_df["GT"] - values_df["estim"])

        global_log_diff = np.histogram(values_df["log_GT"] - values_df["log_estim"])

        mean_diff_per_px = error_map(values_df["diff"])
        mean_log_diff_per_px = error_map(values_df["logdiff"])

        per_fpv = values_df["diff"].groupby(by=np.round(["fpv_dist"])).mean()

        log_diff_per_px = values_df["logdiff"].groupby(by=["x", "y"]).mean()
        log_error_map = np.full(estimated_depth.shape[:2], np.NaN)
        log_error_map[log_diff_per_px.index] = log_diff_per_px

        log_per_fpv = values_df["logdiff"].groupby(by=np.round(["fpv_dist"])).mean()

        fig, axes = plt.subplots(len(estim_per_GT), 2, figsize=(15, 20), dpi=200)
        for i, (k, v) in enumerate(estim_per_GT.items()):

            plot_distribution(axes[i, 0], v["normal"])
            plot_distribution(axes[i, 1], v["log_normal"])
            axes[i, 0].set_title("dstribution of estimation around GT = {:.2f}".format(k))
            axes[i, 1].set_title("dstribution of log estimation around log GT = {:.2f}".format(np.log(k)))

        fig, axes = plt.subplots(2, 1, figsize=(15, 20), dpi=200)
        plot_distribution(axes[0, 0], global_diff)
        axes[0, 0].set_title("Global difference distribution from GT")
        plot_distribution(axes[1, 0], global_log_diff)
        axes[1, 0].set_title("Global log difference distribution from GT")

        fig, axes = plt.subplots(2, 1, figsize=(15, 20), dpi=200)
        plot_distribution(axes[0, 0], per_fpv)
        axes[0, 0].set_title("Mean abs error wrt to distance to fpv (in px)")
        plot_distribution(axes[1, 0], log_per_fpv)
        axes[0, 0].set_title("Mean abs log error wrt to distance to fpv (in px)")

        fig, axes = plt.subplots(2, 1, figsize=(15, 20), dpi=200)
        axes[0, 0].imshow(mean_diff_per_px)
        axes[0, 0].set_title("Mean error for each pixel")
        axes[1, 0].imshow(mean_log_diff_per_px)
        axes[1, 0].set_title("Mean Log error for each pixel")

        plt.show()

    error_names = ["AbsDiff", "StdDiff", "AbsRel", "StdRel", "AbsLog", "StdLog", "a1", "a2", "a3"]
    errors = [values_df["diff"].mean(),
              np.sqrt(np.power(values_df["diff"], 2).mean()),
              values_df["reldiff"].mean(),
              np.sqrt(np.power(values_df["reldiff"], 2).mean()),
              values_df["logdiff"].mean(),
              sum(values_df["log_diff"] < np.log(1.25)) / len(values_df),
              sum(values_df["log_diff"] < 2 * np.log(1.25)) / len(values_df),
              sum(values_df["log_diff"] < 3 * np.log(1.25)) / len(values_df)]
    print("Results for usual metrics")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors))


if __name__ == '__main__':
    main()
