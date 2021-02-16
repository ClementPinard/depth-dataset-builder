from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

parser = ArgumentParser(
    description="Evaluate depth maps with respect to ground truth depth and FPV position",
    formatter_class=ArgumentDefaultsHelpFormatter,
)

parser.add_argument("--dataset_root", metavar="DIR", type=Path)
parser.add_argument(
    "--est_depth",
    metavar="PATH",
    type=Path,
    nargs='+',
    help="where the estimated depth maps are stored, must be a 3D npz file",
)
parser.add_argument(
    "--algorithm_names",
    "--names",
    metavar="STR",
    type=str,
    nargs='+',
    help="name of the algorithms corresponding to estimated depth arrays")

parser.add_argument(
    "--evaluation_list_path",
    "--eval",
    metavar="PATH",
    type=Path,
    help="File with list of images to test for depth evaluation",
)
parser.add_argument(
    "--flight_path_vector_list",
    "--fpv",
    metavar="PATH",
    type=Path,
    help="File with list of speed vectors, used to compute error wrt direction",
)
parser.add_argument(
    "--scale_invariant",
    action="store_true",
    help="If selected, will rescale depth map with ratio of medians",
)
parser.add_argument(
    "--min_depth",
    metavar="D",
    default=1e-2,
    type=float,
    help="threshold below which GT is discarded",
)
parser.add_argument(
    "--max_depth",
    metavar="D",
    default=250,
    type=float,
    help="threshold above which GT is discarded",
)
parser.add_argument(
    "--depth_mask",
    metavar="PATH",
    default=None,
    type=Path,
    help="path to boolean numpy array. Should be the same size as ground truth. "
    "False value will discard the corresponding pixel location for every ground truth",
)

parser.add_argument(
    "--output_figures",
    metavar="DIR",
    default=None,
    type=Path,
    help="where to save the figures, in pgf format. If not set, will show them with plt.show()"
)

coords = None


def get_values(
    gt_depth, estim_depth, fpv, scale_invariant=False, mask=None, min_depth=1e-2, max_depth=250
):
    """Given a depth maps and depth estimation, return a table of all valid depth points with
    additional metadata

    Args:
        gt_depth (np.array): ground truth depth computed by RDC
        estim_depth (np.array): Depth estimated with inference toolkit
        fpv (np.array): array of 2 floats, representing the fpv coordinates, in pixels
        scale_invariant (bool, optional): If set to True, will multiply estimated depth with
                                          ratio between medians. This is representative of how
                                          depth was evaluated in Eigen et al.
        mask (np.array, optional): Boolean array of same shape as depth maps. Discard from evaluation
                                   image points where mask[u,v] == False
        min_depth (float, optional): Minimal depth below which ground truth is discarded and estimation is clipped.*
                                     Defaults to 1e-2.
        max_depth (float, optional): Maximal depth above which ground truth is discarded estimation is clipped.
                                     Defaults to 250.

    Returns:
        [type]: [description]
    """
    global coords
    if coords is None:
        coords = np.stack(
            np.meshgrid(np.arange(gt_depth.shape[1]), np.arange(gt_depth.shape[0])), axis=-1
        )

    # TODO : For now, fpv distance is given in pixel distance.
    # A more accurate way would be to use angular distance.
    fpv_dist = np.linalg.norm(coords - fpv, axis=-1)
    estim_depth = np.clip(estim_depth, min_depth, max_depth)
    valid = (gt_depth > min_depth) & (gt_depth < max_depth)
    if mask is not None:
        valid = valid & mask
    if valid.sum() == 0:
        return
    valid_gt, valid_estim = gt_depth[valid], estim_depth[valid]
    if scale_invariant:
        valid_estim = valid_estim * np.median(valid_gt) / np.median(valid_estim)
    fpv_dist = fpv_dist[valid]
    valid_coords = coords[valid]
    values = np.stack([valid_gt, valid_estim, *valid_coords.T, fpv_dist], axis=-1)

    return pd.DataFrame(values, columns=["GT", "estim", "x", "y", "fpv_dist"])


def plot_distribution(values, bins, ax, label=None, log_bins=False):
    """Distribution plotting function, will plot with lines instead of bars

    Args:
        values: histogram to plot
        bins: Corresponding bins delimitting histogram values
        ax: Matplotlib ax to plot on
        label (str): Plot label name. Defaults to None.
        log_bins (bool): If set to True, will set the scale to log.
                         Useful for Mean Log Error. Defaults to False.
    """
    bin_dists = bins[1:] - bins[:-1]
    total = sum(bin_dists)
    normalized_values = (values / sum(values)) * bin_dists / total
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    if log_bins:
        bin_centers = np.exp(bin_centers)
    ax.plot(bin_centers, normalized_values, label=label)
    if log_bins:
        ax.set_xscale("log")


def group_quantiles(df, to_group, columns, quantiles=[0.25, 0.5, 0.75]):
    if isinstance(columns, str):
        columns = [columns]
    grouped_df = df.groupby(by=np.round(df[to_group]))
    return grouped_df[columns].quantile(quantiles).unstack()


def error_map(error_per_px):
    """Compute Image with pixelwise mean depth error

    Args:
        error_per_px (pd.Series): Table of errors with pixels as index

    Returns:
        np.array: Array with the shape of the image, with mean error at each pixel
    """
    x, y = np.stack(error_per_px.index.values, axis=-1).astype(int)
    error_map = np.full((int(x.max() + 1), int(y.max() + 1)), np.NaN)
    error_map[x, y] = error_per_px.values
    return error_map


def error_metrics(df, algo_name, suffix=''):
    error_names = ["AbsDiff", "StdDiff", "AbsRel", "StdRel", "AbsLog", "StdLog", "a1", "a2", "a3"]
    errors = [
        df["absdiff"].mean(),
        np.sqrt(df["absdiff2"].mean()),
        df["reldiff"].mean(),
        np.sqrt(df["reldiff2"].mean()),
        df["abslogdiff"].mean(),
        np.sqrt(df["logdiff2"].mean()),
        df["a1"].mean(),
        df["a2"].mean(),
        df["a3"].mean(),
    ]

    # Print the results
    # TODO : save the result in latex tab format ?
    print("Results for usual metrics for algorithm {}, {}".format(algo_name, suffix))
    print(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format(
            *error_names
        )
    )
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
            *errors
        )
    )


def main():
    args = parser.parse_args()
    assert (len(args.est_depth) == len(args.algorithm_names))

    if args.output_figures is not None:
        matplotlib.use("pgf")
        pgf_with_xelatex = {
            'text.usetex': True,
            "pgf.texsystem": "xelatex",
            "pgf.preamble": r"\usepackage{amssymb} "
                            r"\usepackage{amsmath} "
                            r"\usepackage{fontspec} "
                            r"\usepackage{unicode-math}"
        }
        # Change to pgf if needed
        savefig_ext = "pdf"
        matplotlib.rcParams.update(pgf_with_xelatex)

    with open(args.evaluation_list_path, "r") as f:
        depth_paths = [line[:-1] for line in f.readlines()]
    fpv_list = np.loadtxt(args.flight_path_vector_list)
    dataframes = {}
    for p, name in zip(args.est_depth, args.algorithm_names):
        estimated_depth = np.load(p, allow_pickle=True)
        values_df = []
        assert len(depth_paths) == len(estimated_depth)
        if args.depth_mask is not None:
            mask = np.load(args.depth_mask)
        else:
            mask = None

        # Load each GT-estimation pair and extract data in a pandas dataframe
        # values_df is at first a list of dataframes which we then concatenate
        print("getting results for {} algorithm (file : {})".format(name, p))
        for filepath, fpv in tqdm(zip(depth_paths, fpv_list), total=len(fpv_list)):
            GT = np.load((args.dataset_root / filepath).stripext() + ".npy")
            new_values = get_values(
                GT,
                estimated_depth[filepath],
                fpv,
                args.scale_invariant,
                mask,
                args.min_depth,
                args.max_depth,
            )
            if new_values is not None:
                values_df.append(new_values)
        values_df = pd.concat(values_df)

        # Additional values to the Dataframe
        # Note that no mean is computed here, each row in the dataframe is ONE pixel
        # The dataframe is thus potentially thousands rows long
        values_df["log_GT"] = np.log(values_df["GT"])
        values_df["log_estim"] = np.log(values_df["estim"])
        values_df["diff"] = values_df["estim"] - values_df["GT"]
        values_df["absdiff"] = values_df["diff"].abs()
        values_df["absdiff2"] = np.power(values_df["diff"], 2)
        values_df["reldiff"] = values_df["absdiff"] / values_df["GT"]
        values_df["reldiff2"] = np.power(values_df["reldiff"], 2)
        values_df["logdiff"] = values_df["log_estim"] - values_df["log_GT"]
        values_df["logdiff2"] = np.power(values_df["absdiff"], 2)
        values_df["abslogdiff"] = values_df["logdiff"].abs()
        values_df["a1"] = (values_df["abslogdiff"] < np.log(1.25)).astype(float)
        values_df["a2"] = (values_df["abslogdiff"] < 2 * np.log(1.25)).astype(float)
        values_df["a3"] = (values_df["abslogdiff"] < 3 * np.log(1.25)).astype(float)
        dataframes[name] = values_df

    for name, df in dataframes.items():
        print()
        print("---------------------------")
        print("Results for {}".format(name))
        print("---------------------------")
        print()
        # Compute mean erros, a la Eigen et al.
        error_metrics(df, name, "averaged over all points")
        # Get mean values per ground truth values, and then mean them
        # This way, we have the same weight for each ground truth value
        values_df_per_gt = df.groupby(by=np.round(values_df["GT"])).mean()
        error_metrics(values_df_per_gt, name, "averaged over gt values")
        values_df_per_log_gt = df.groupby(by=0.1 * np.round(10 * values_df["log_GT"])).mean()
        error_metrics(values_df_per_log_gt, name, "averaged over log(gt) values")

    # TODO better handling of the parameters, maybe just add it to argparse
    plot = True
    n_bins = 4
    if plot:
        # COMPUTING HISTOGRAMS

        # GT-wise difference with estimation distributions.
        # Useful to see if we perform well
        # in the depth range we are actually interested in
        # Construct the bins for GT-wise error
        # You can see this as 3D histogram
        # Note that if the assumptions of gaussian difference
        # for log values, the log_normal error should be roughly
        # the same for all bins

        min_gt = values_df["GT"].min()
        max_gt = values_df["GT"].max()
        bins = np.linspace(min_gt, max_gt, n_bins + 1)

        histograms = {}
        for name, df in dataframes.items():
            histograms[name] = {}
            estim_per_GT = {}
            for b1, b2 in zip(bins[:-1], bins[1:]):
                per_gt = df[(df["GT"] > b1) & (df["GT"] < b2)]
                estim_per_GT[(b1 + b2) / 2] = {
                    "normal": np.histogram(per_gt["diff"], bins=100),
                    "log_normal": np.histogram(per_gt["logdiff"], bins=100),
                    "bins": [b1, b2],
                }
            histograms[name]["estim_per_GT"] = estim_per_GT

            # Global histograms
            # Same as above, but with one bin, and thus not GT-wise

            histograms[name]["global_diff"] = np.histogram(df["estim"] - df["GT"], bins=100)
            histograms[name]["global_log_diff"] = np.histogram(df["log_estim"] - df["log_GT"], bins=100)

            # Depth error per pixel
            # Useful to identify if a region in the screen is particualrly faulty.
            # Can help spot dataset inconsistency (eg sky is always in the same place)
            # Can also help find calibration artefacts ?

            metric_per_px = df.groupby(by=["x", "y"]).mean()
            histograms[name]["mean_diff_per_px"] = error_map(metric_per_px["absdiff"])
            histograms[name]["mean_log_diff_per_px"] = error_map(metric_per_px["logdiff"])

            # Depth error wrt pixelwise distance to FPV. For SFM, the closer we are to FPV,
            # The harde it is to deduce depth. But in the same time, the more
            # usefule depth becomes, because it indicates distances of obstacles where
            # we are headed to.

            # Note : if fpv is too far, it means it is not on the image
            # And thus this metric is not really interesting.
            histograms[name]["quantiles_per_fpv"] = group_quantiles(
                df[df["fpv_dist"] < 1000], "fpv_dist", ["absdiff", "abslogdiff"]
            )
            histograms[name]["quantiles_per_gt"] = group_quantiles(df, "GT", ["absdiff", "abslogdiff"])
            histograms[name]["quantiles_per_estimation"] = group_quantiles(df, "estim", ["absdiff", "abslogdiff"])

        # PLOTTING
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # First plot, general insight for dataset
        fig1, axes = plt.subplots(2, 1, sharex=True)
        GT_distrib = None
        for name, df in dataframes.items():
            if GT_distrib is None:
                GT_distrib = np.histogram(df["GT"], bins=100)
                plot_distribution(*GT_distrib, axes[0], label="groundtruth depth")
            estim_distrib = np.histogram(df["estim"], bins=100)
            plot_distribution(*estim_distrib, axes[1], label=name)
        axes[0].set_title("Ground Truth distribution")
        axes[0].legend()
        axes[1].set_title("depth estimation distribution from {}".format(name))
        axes[1].legend()
        if args.output_figures is not None:
            fig1.savefig(args.output_figures / "depth_distrib.{}".format(savefig_ext))

        # Second plot, GT-wise difference one figure per algorithm
        for name, h in histograms.items():
            fig2, axes = plt.subplots(1, 2, sharey=True)
            for i, (k, v) in enumerate(h["estim_per_GT"].items()):
                plot_distribution(
                    *v["normal"], axes[0], label="$GT \\in [{:.1f},{:.1f}]$".format(*v["bins"])
                )
                plot_distribution(
                    *v["log_normal"],
                    axes[1],
                    label="$GT \\in [{:.1f},{:.1f}]$".format(*v["bins"]),
                    log_bins=True
                )
                # axes[0, 0].set_title("distribution of estimation around GT = {:.2f}".format(k))
                # axes[0, 1].set_title("distribution of log estimation around log GT = {:.2f}".format(np.log(k)))
            axes[0].legend()
            axes[0].set_title("GT - estimation difference")
            axes[1].legend()
            axes[1].set_title("logt GT - log estimation difference")
            fig2.tight_layout()
            if args.output_figures is not None:
                fig2.savefig(args.output_figures / "GTwise_depth_diff_{}.{}".format(name, savefig_ext))

        # Third plot, global diff histogram
        fig, axes = plt.subplots(2, 1)
        for name, h in histograms.items():
            plot_distribution(*h["global_diff"], axes[0], name)
            plot_distribution(*h["global_log_diff"], axes[1], name, log_bins=True)
        axes[1].set_title("Global log difference distribution from GT")
        axes[0].set_title("Global difference distribution from GT")
        axes[1].legend()
        axes[0].legend()
        plt.tight_layout()
        if args.output_figures is not None:
            fig.savefig(args.output_figures / "global_depth_diff.{}".format(savefig_ext))

        def plot_quartile(axes, color, algo_name, df):
            index = df.index
            diff = df["absdiff"]
            logdiff = df["abslogdiff"]
            axes[0].fill_between(
                index, diff[0.25], diff[0.75], color=c, alpha=0.1
            )
            axes[0].plot(diff[0.5].index, diff[0.5], color=c, label=algo_name)
            axes[1].fill_between(
                index, logdiff[0.25], logdiff[0.75], color=c, alpha=0.1
            )
            axes[1].plot(logdiff[0.5], label=algo_name)
            axes[0].legend()
            axes[1].legend()

        # Fourth plot, error wrt distance to fpv
        fig, axes = plt.subplots(2, 1, sharex=True)
        for c, (name, h) in zip(colors, histograms.items()):
            plot_quartile(axes, c, name, h["quantiles_per_fpv"])
        axes[0].set_title("Error wrt to distance to fpv (in px)")
        axes[1].set_title("Log error wrt to distance to fpv (in px)")
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Distance to flight path vector (in px)")
        plt.tight_layout()
        if args.output_figures is not None:
            fig.savefig(args.output_figures / "fpv_error_quantiles.{}".format(savefig_ext))

        # Fifth plot, another way of plotting GT-wise error:
        # For each GT depth, we show 3 points : median, and 50% confidence intervale (2 points)
        # We have less info than the full histogram but we can show more GT values
        fig, axes = plt.subplots(2, 1, sharex=True)
        for c, (name, h) in zip(colors, histograms.items()):
            plot_quartile(axes, c, name, h["quantiles_per_gt"])
        axes[0].set_title("Error wrt to groundtruth depth")
        axes[1].set_title("Log error wrt to groundtruth depth")
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Estimated depth (in meters)")
        plt.tight_layout()
        if args.output_figures is not None:
            fig.savefig(args.output_figures / "gt_error_quantiles.{}".format(savefig_ext))

        # Last plot, error with respect to estimated depth

        fig, axes = plt.subplots(2, 1, sharex=True)
        for c, (name, h) in zip(colors, histograms.items()):
            plot_quartile(axes, c, name, h["quantiles_per_estimation"])
        axes[0].set_title("Error wrt to estimated depth")
        axes[1].set_title("Log error wrt to estimated depth")
        axes[1].set_yscale('log')
        axes[1].set_xlabel("Estimated depth (in meters)")
        plt.tight_layout()
        if args.output_figures is not None:
            fig.savefig(args.output_figures / "est_error_quantiles.{}".format(savefig_ext))

        # Last plot, pixelwise error
        for name, h in histograms.items():
            fig, axes = plt.subplots(2, 1)
            pl = axes[0].imshow(h["mean_diff_per_px"].T)
            axes[0].set_title("Mean error for each pixel")
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(pl, cax=cax)
            cbar.ax.tick_params(axis="y", direction="in")
            pl = axes[1].imshow(h["mean_log_diff_per_px"].T)
            axes[1].set_title("Mean Log error for each pixel")
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(pl, cax=cax)
            cbar.ax.tick_params(axis="y", direction="in")
            plt.tight_layout()
            if args.output_figures is not None:
                fig.savefig(args.output_figures / "pixel_error_map_{}.{}".format(name, savefig_ext))
        if args.output_figures is None:
            plt.show()


if __name__ == "__main__":
    main()
