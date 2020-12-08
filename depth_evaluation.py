from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

parser = ArgumentParser(description='Convert EuroC dataset to COLMAP',
                        formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset_root', metavar='DIR', type=Path)
parser.add_argument('--est_depth', metavar='DIR', type=Path,
                    help='where the depth maps are stored, must be a 3D npy file')
parser.add_argument('--evaluation_list_path', metavar='PATH', type=Path,
                    help='File with list of images to test for depth evaluation')
parser.add_argument('--flight_path_vector_list', metavar='PATH', type=Path,
                    help='File with list of speed vectors, used to compute error wrt direction')
parser.add_argument('--scale-invariant', action='store_true',
                    help='If selected, will rescale depth map with ratio of medians')

coords = None


def get_values(gt_depth, estim_depth, fpv, min_depth=1e-2, max_depth=250):
    global coords
    if coords is None:
        coords = np.stack(np.meshgrid(np.arange(gt_depth.shape[1]), np.arange(gt_depth.shape[0])), axis=-1)
    fpv_dist = np.linalg.norm(coords - fpv, axis=-1)
    estim_depth = np.clip(estim_depth, min_depth, max_depth)
    valid = (gt_depth > min_depth) & (gt_depth < max_depth)
    fpv_dist = fpv_dist[valid]
    valid_coords = coords[valid]
    values = np.stack([gt_depth[valid], estim_depth[valid], *valid_coords.T, fpv_dist], axis=-1)

    return pd.DataFrame(values, columns=["GT", "estim", "x", "y", "fpv_dist"])


def plot_distribution(values, bins, ax, label=None, log_bins=False):
    bin_dists = bins[1:] - bins[:-1]
    total = sum(bin_dists)
    normalized_values = (values / sum(values)) * bin_dists / total
    bin_centers = 0.5*(bins[1:] + bins[:-1])
    if log_bins:
        bin_centers = np.exp(bin_centers)
    ax.plot(bin_centers, normalized_values, label=label)
    if log_bins:
        ax.set_xscale('log')


def group_quantiles(df, to_group, columns, quantiles=[0.25, 0.5, 0.75]):
    if isinstance(columns, str):
        columns = [columns]
    grouped_df = df.groupby(by=np.round(df[to_group]))
    return grouped_df[columns].quantile(quantiles).unstack()


def error_map(error_per_px):
    x, y = np.stack(error_per_px.index.values, axis=-1).astype(int)
    error_map = np.full((int(x.max() + 1), int(y.max() + 1)), np.NaN)
    error_map[x, y] = error_per_px.values
    return error_map


def main():
    args = parser.parse_args()
    n_bins = 4
    with open(args.evaluation_list_path, 'r') as f:
        depth_paths = [line[:-1] for line in f.readlines()]
    fpv_list = np.loadtxt(args.flight_path_vector_list)
    estimated_depth = np.load(args.est_depth, allow_pickle=True)
    values_df = []
    assert(len(depth_paths) == len(estimated_depth))
    for filepath, fpv in zip(depth_paths, tqdm(fpv_list)):
        GT = np.load(args.dataset_root/filepath + '.npy')
        new_values = get_values(GT, estimated_depth[filepath], fpv)
        values_df.append(new_values)

    values_df = pd.concat(values_df)
    values_df["log_GT"] = np.log(values_df["GT"])
    values_df["log_estim"] = np.log(values_df["estim"])
    values_df["diff"] = values_df["estim"] - values_df["GT"]
    values_df["absdiff"] = values_df["diff"].abs()
    values_df["reldiff"] = values_df["diff"] / values_df["GT"]
    values_df["logdiff"] = values_df["log_estim"] - values_df["log_GT"]
    values_df["abslogdiff"] = values_df["logdiff"].abs()

    error_names = ["AbsDiff", "StdDiff", "AbsRel", "StdRel", "AbsLog", "StdLog", "a1", "a2", "a3"]
    errors = [values_df["diff"].mean(),
              np.sqrt(np.power(values_df["diff"], 2).mean()),
              values_df["reldiff"].mean(),
              np.sqrt(np.power(values_df["reldiff"], 2).mean()),
              values_df["logdiff"].abs().mean(),
              np.sqrt(np.power(values_df["logdiff"], 2).mean()),
              sum(values_df["logdiff"] < np.log(1.25)) / len(values_df),
              sum(values_df["logdiff"] < 2 * np.log(1.25)) / len(values_df),
              sum(values_df["logdiff"] < 3 * np.log(1.25)) / len(values_df)]
    print("Results for usual metrics")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors))

    plot = True
    if plot:

        min_gt = values_df["GT"].min()
        max_gt = values_df["GT"].max()

        bins = np.linspace(min_gt, max_gt, n_bins + 1)

        estim_per_GT = {}
        for b1, b2 in zip(bins[:-1], bins[1:]):
            per_gt = values_df[(values_df["GT"] > b1) & (values_df["GT"] < b2)]
            estim_per_GT[(b1+b2)/2] = {"normal": np.histogram(per_gt["diff"], bins=100),
                                       "log_normal": np.histogram(per_gt["logdiff"], bins=100),
                                       "bins": [b1, b2]}

        global_diff = np.histogram(values_df["estim"] - values_df["GT"], bins=100)

        global_log_diff = np.histogram(values_df["log_estim"] - values_df["log_GT"], bins=100)

        metric_per_px = values_df.groupby(by=["x", "y"]).mean()
        mean_diff_per_px = error_map(metric_per_px["absdiff"])
        mean_log_diff_per_px = error_map(metric_per_px["logdiff"])

        quantiles_per_fpv = group_quantiles(values_df[values_df["fpv_dist"] < 1000],
                                            "fpv_dist",
                                            ["absdiff", "abslogdiff"])
        quantiles_per_gt = group_quantiles(values_df, "GT", ["absdiff", "abslogdiff"])

        # metric_per_gt = values_df.groupby(by = np.round(values_df["GT"]))

        fig, axes = plt.subplots(2, 1, sharex=True)
        GT_distrib = np.histogram(values_df["GT"], bins=100)
        plot_distribution(*GT_distrib, axes[0])
        axes[0].set_title("Ground Truth distribution")
        estim_distrib = np.histogram(values_df["estim"], bins=100)
        plot_distribution(*estim_distrib, axes[1])
        axes[1].set_title("depth estimation distribution")

        fig, axes = plt.subplots(1, 2, sharey=True)
        for i, (k, v) in enumerate(estim_per_GT.items()):
            plot_distribution(*v["normal"], axes[0], label="$GT \\in [{:.1f},{:.1f}]$".format(*v["bins"]))
            plot_distribution(*v["log_normal"], axes[1], label="$GT \\in [{:.1f},{:.1f}]$".format(*v["bins"]), log_bins=True)
            axes[0].legend()
            axes[1].legend()
            # axes[0, 0].set_title("dstribution of estimation around GT = {:.2f}".format(k))
            # axes[0, 1].set_title("dstribution of log estimation around log GT = {:.2f}".format(np.log(k)))

        fig, axes = plt.subplots(2, 1)
        plot_distribution(*global_diff, axes[0])
        axes[0].set_title("Global difference distribution from GT")
        plot_distribution(*global_log_diff, axes[1], log_bins=True)
        axes[1].set_title("Global log difference distribution from GT")

        plt.tight_layout()
        fig, axes = plt.subplots(2, 1, sharex=True)
        index = quantiles_per_fpv.index
        diff_per_fpv = quantiles_per_fpv["absdiff"]
        logdiff_per_fpv = quantiles_per_fpv["abslogdiff"]
        axes[0].fill_between(index, diff_per_fpv[0.25], diff_per_fpv[0.75], color="cyan")
        axes[0].plot(diff_per_fpv[0.5].index, diff_per_fpv[0.5])
        axes[0].set_title("Error wrt to distance to fpv (in px)")
        axes[1].fill_between(index, logdiff_per_fpv[0.25], logdiff_per_fpv[0.75], color="cyan")
        axes[1].plot(logdiff_per_fpv[0.5])
        axes[1].set_title("Log error wrt to distance to fpv (in px)")

        plt.tight_layout()
        fig, axes = plt.subplots(2, 1, sharex=True)
        index = quantiles_per_gt.index
        diff_per_gt = quantiles_per_gt["absdiff"]
        logdiff_per_gt = quantiles_per_gt["abslogdiff"]
        axes[0].fill_between(index, diff_per_gt[0.25], diff_per_gt[0.75], color="cyan")
        axes[0].plot(diff_per_gt[0.5])
        axes[0].set_title("Error wrt to distance to groundtruth depth")
        axes[1].fill_between(index, logdiff_per_gt[0.25], logdiff_per_gt[0.75], color="cyan")
        axes[1].plot(logdiff_per_gt[0.5])
        axes[1].set_title("Log error wrt to groundtruth depth")

        plt.tight_layout()
        fig, axes = plt.subplots(2, 1)
        axes[0].imshow(mean_diff_per_px.T)
        axes[0].set_title("Mean error for each pixel")
        axes[1].imshow(mean_log_diff_per_px.T)
        axes[1].set_title("Mean Log error for each pixel")
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
