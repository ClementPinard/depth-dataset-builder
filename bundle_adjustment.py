# imports
import numpy as np
import pandas as pd
import torch
from pytorch3d.transforms.so3 import so3_exponential_map, so3_relative_angle, so3_log_map
from pyntcloud import PyntCloud
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures.utils import list_to_padded
from pytorch3d.loss import chamfer_distance
from colmap_util.read_model import read_model, qvec2rotmat
# add path for demo utils
import sys
import os
from utils import plot_camera_scene
from torch.nn.functional import smooth_l1_loss
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from path import Path

sys.path.append(os.path.abspath(''))
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = ArgumentParser(description='Perform Bundle Adjustment on COLMAP sparse model',
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-i', '--input_folder', type=Path, required=True,
                    help='input colmap model')
parser.add_argument('--ply', type=Path, required=True, help='PLY model to apply chamfer loss to')
parser.add_argument('-o', '--output_folder', type=Path, required=True,
                    help='output colmap model')
parser.add_argument('--dtype', default='float', choices=['float', 'double'])
parser.add_argument('--lr', help='learning rate', default=0.1, type=float)


def main(args):
    # set for reproducibility
    torch.manual_seed(42)
    if args.dtype == "float":
        args.dtype = torch.float32
    elif args.dtype == "double":
        args.dtype = torch.float64

    # ## 1. Set up Cameras and load ground truth positions

    # load the SE3 graph of relative/absolute camera positions
    if (args.input_folder / "images.bin").isfile():
        ext = '.bin'
    elif (args.input_folder / "images.txt").isfile():
        ext = '.txt'
    else:
        print('error')
        return
    cameras, images, points3D = read_model(args.input_folder, ext)

    images_df = pd.DataFrame.from_dict(images, orient="index").set_index("id")
    cameras_df = pd.DataFrame.from_dict(cameras, orient="index").set_index("id")
    points_df = pd.DataFrame.from_dict(points3D, orient="index").set_index("id")
    print(points_df)
    print(images_df)
    print(cameras_df)

    ref_pointcloud = PyntCloud.from_file(args.ply)

    points_3d = np.stack(points_df["xyz"].values)
    points_3d = torch.from_numpy(points_3d).to(device, dtype=args.dtype)

    cameras_R = np.stack([qvec2rotmat(q) for _, q in images_df["qvec"].iteritems()])
    cameras_R = torch.from_numpy(cameras_R).to(device, dtype=args.dtype).transpose(1, 2)

    cameras_T = torch.from_numpy(np.stack(images_df["tvec"].values)).to(device, dtype=args.dtype)

    cameras_params = torch.from_numpy(np.stack(cameras_df["params"].values)).to(device, dtype=args.dtype)
    cameras_params = cameras_params[:, :4]
    print(cameras_params)

    # Constructu visibility map, True at (frame, point) if point is visible by frame, False otherwise
    # Thus, we can ignore reprojection errors for invisible points
    visibility = np.full((cameras_R.shape[0], points_3d.shape[0]), False)
    visibility = pd.DataFrame(visibility, index=images_df.index, columns=points_df.index)

    points_2D_gt = []
    for idx, (pts_ids, xy) in images_df[["point3D_ids", "xys"]].iterrows():
        pts_ids_clean = pts_ids[pts_ids != -1]
        pts_2D = pd.DataFrame(xy[pts_ids != -1], index=pts_ids_clean)
        pts_2D = pts_2D[~pts_2D.index.duplicated(keep=False)].reindex(points_df.index).dropna()
        points_2D_gt.append(pts_2D.values)
        visibility.loc[idx, pts_2D.index] = True

    print(visibility)

    visibility = torch.from_numpy(visibility.values).to(device)
    eps = 1e-3
    # Visibility map is very sparse. So we can use Pytorch3d's function to reduce points_2D size
    # to (num_frames, max points seen by frame)
    points_2D_gt = list_to_padded([torch.from_numpy(p) for p in points_2D_gt], pad_value=eps).to(device, dtype=args.dtype)
    print(points_2D_gt)

    cameras_df["raw_id"] = np.arange(len(cameras_df))
    cameras_id_per_image = torch.from_numpy(cameras_df["raw_id"][images_df["camera_id"]].values).to(device)
    # the number of absolute camera positions
    N = len(images_df)
    nonzer = (points_2D_gt != eps).all(dim=-1)
    # print(padded)
    # print(points_2D_gt, points_2D_gt.shape)

    # ## 2. Define optimization functions
    #
    # ### Relative cameras and camera distance
    # We now define two functions crucial for the optimization.
    #
    # **`calc_camera_distance`** compares a pair of cameras.
    # This function is important as it defines the loss that we are minimizing.
    # The method utilizes the `so3_relative_angle` function from the SO3 API.
    #
    # **`get_relative_camera`** computes the parameters of a relative camera
    # that maps between a pair of absolute cameras. Here we utilize the `compose`
    # and `inverse` class methods from the PyTorch3D Transforms API.

    def calc_camera_distance(cam_1, cam_2):
        """
        Calculates the divergence of a batch of pairs of cameras cam_1, cam_2.
        The distance is composed of the cosine of the relative angle between
        the rotation components of the camera extrinsics and the l2 distance
        between the translation vectors.
        """
        # rotation distance
        R_distance = (1.-so3_relative_angle(cam_1.R, cam_2.R, cos_angle=True)).mean()
        # translation distance
        T_distance = ((cam_1.T - cam_2.T)**2).sum(1).mean()
        # the final distance is the sum
        return R_distance + T_distance

    # ## 3. Optimization
    # Finally, we start the optimization of the absolute cameras.
    #
    # We use SGD with momentum and optimize over `log_R_absolute` and `T_absolute`.
    #
    # As mentioned earlier, `log_R_absolute` is the axis angle representation of the
    # rotation part of our absolute cameras. We can obtain the 3x3 rotation matrix
    # `R_absolute` that corresponds to `log_R_absolute` with:
    #
    # `R_absolute = so3_exponential_map(log_R_absolute)`
    #

    fxfyu0v0 = cameras_params[cameras_id_per_image]
    cameras_absolute_gt = PerspectiveCameras(
        focal_length=fxfyu0v0[:, :2],
        principal_point=fxfyu0v0[:, 2:],
        R=cameras_R,
        T=cameras_T,
        device=device,
    )
    with torch.no_grad():
        padded_points = list_to_padded([points_gt[visibility[c]] for c in range(N)], pad_value=1e-3)
        points_2D_gt = cameras_absolute_gt.transform_points(padded_points, eps=1e-4)[:, :, :2]

    # initialize the absolute log-rotations/translations with random entries
    # log_R_absolute_init = torch.randn(N, 3, dtype=torch.float32, device=device)
    # T_absolute_init = torch.randn(N, 3, dtype=torch.float32, device=device)
    noise = 0
    shift = 0.1
    points_absolute_init = points_gt + noise*torch.randn(points_gt.shape, dtype=torch.float32, device=device) + shift

    log_R_absolute_init = so3_log_map(cameras_R) + noise * torch.randn(N, 3, dtype=torch.float32, device=device)
    T_absolute_init = cameras_T + noise * torch.randn(cameras_T.shape, dtype=torch.float32, device=device) - shift
    cams_init = cameras_params + noise * torch.randn(cameras_params.shape, dtype=torch.float32, device=device)
    # points_absolute_init = points_gt

    # furthermore, we know that the first camera is a trivial one
    #    (see the description above)
    #log_R_absolute_init[0, :] = 0.
    #T_absolute_init[0, :] = 0.

    # instantiate a copy of the initialization of log_R / T
    log_R_absolute = log_R_absolute_init.clone().detach()
    log_R_absolute.requires_grad = True
    T_absolute = T_absolute_init.clone().detach()
    T_absolute.requires_grad = True

    cams_params = cams_init.clone().detach()
    cams_params.requires_grad = True

    points_absolute = points_absolute_init.clone().detach()
    points_absolute.requires_grad = True

    # init the optimizer
    optimizer = torch.optim.SGD([points_absolute, log_R_absolute, T_absolute], lr=args.lr, momentum=0.9)

    # run the optimization
    n_iter = 200000  # fix the number of iterations
    with torch.no_grad():
        padded_points = list_to_padded([points_gt[visibility[c]] for c in range(N)], pad_value=1e-3)
        projected_points = cameras_absolute_gt.transform_points(padded_points, eps=1e-4)[:, :, :2]
        points_distance = ((projected_points[nonzer] - points_2D_gt[nonzer]) ** 2).sum(dim=1)
        inliers = (points_distance < 100).clone().detach()
        print(inliers)
    loss_log = []
    cam_dist_log = []
    pts_dist_log = []
    for it in range(n_iter):
        # re-init the optimizer gradients
        optimizer.zero_grad()

        # compute the absolute camera rotations as
        # an exponential map of the logarithms (=axis-angles)
        # of the absolute rotations
        # R_absolute = so3_exponential_map(log_R_absolute)
        R_absolute = cameras_R

        fxfyu0v0 = cams_params[cameras_id_per_image]
        # get the current absolute cameras
        cameras_absolute = PerspectiveCameras(
                focal_length=fxfyu0v0[:, :2],
                principal_point=fxfyu0v0[:, 2:],
            R=R_absolute,
            T=T_absolute,
            device=device,
        )

        padded_points = list_to_padded([points_absolute[visibility[c]] for c in range(N)], pad_value=1e-3)

        projected_points_3D = cameras_absolute.transform_points(padded_points, eps=1e-4)
        projected_points = projected_points_3D[:, :, :2]
        with torch.no_grad():
            inliers = inliers & (projected_points_3D[:, :, 2][nonzer] > 0)
        #print(R_absolute[0], cameras_R[0])
        #print(T_absolute[0], cameras_T[0])
        #print(projected_points[0,0], points_2D_gt[0,0])
        # distances = (projected_points[0] - points_2D_gt[0]).norm(dim=-1).detach().cpu().numpy()
        # from matplotlib import pyplot as plt
        # plt.plot(distances[:(visibility[0]).sum()])

        # compare the composed cameras with the ground truth relative cameras
        # camera_distance corresponds to $d$ from the description
        # points_distance = smooth_l1_loss(projected_points, points_2D_gt)
        # points_distance = (smooth_l1_loss(projected_points, points_2D_gt, reduction='none')[nonzer]).sum(dim=1)
        points_distance = ((projected_points[nonzer] - points_2D_gt[nonzer]) ** 2).sum(dim=1)
        points_distance_filtered = points_distance[inliers]
        # 100*((points_gt - points_absolute) ** 2).sum() #
        loss = points_distance_filtered.mean() + 10000*chamfer_distance(points_gt[None], points_absolute[None])[0]
        # our loss function is the camera_distance

        loss.backward()
        # print("faulty elements :")
        # faulty_points = torch.arange(points_absolute.shape[0])[points_absolute.grad[:, 0] != points_absolute.grad[:, 0]]
        # # faulty_images = torch.arange(log_R_absolute.shape[0])[log_R_absolute.grad[:, 0] != log_R_absolute.grad[:, 0]]
        # faulty_cams = torch.arange(cams_params.shape[0])[cams_params.grad[:, 0] != cams_params.grad[:, 0]]
        # print(torch.isnan(projected_points.grad).any(dim=2))
        # print(projected_points.grad.shape)
        # faulty_projected_points = torch.arange(projected_points.shape[1])[torch.isnan(projected_points.grad).any(dim=2)[0]]
        # # print(log_R_absolute[faulty_images])
        # # print(T_absolute[faulty_images])
        # print(points_gt[faulty_points])
        # print(cams_params[faulty_cams])
        # print(projected_points[torch.isnan(projected_points.grad)])
        # first_faulty_point = points_df.iloc[int(faulty_points[0])]
        # faulty_images = images_df.loc[first_faulty_point["image_ids"][0]]
        # print(first_faulty_point)
        # print(faulty_images)

        # apply the gradients
        optimizer.step()

        # plot and print status message
        if it % 2000 == 0 or it == n_iter-1:
            camera_distance = calc_camera_distance(cameras_absolute, cameras_absolute_gt)
            points_3d_distance = (points_gt - points_absolute).norm(dim=-1).mean()
            print('iteration=%3d; loss=%1.3e, points_distance=%1.3e, camera_distance=%1.3e' % (it, loss, points_3d_distance, camera_distance))
            loss_log.append(loss.item())
            pts_dist_log.append(points_3d_distance.item())
            cam_dist_log.append(camera_distance.item())
        if it % 20000 == 0 or it == n_iter-1:
            with torch.no_grad():
                from matplotlib import pyplot as plt
                plt.hist(torch.sqrt(points_distance_filtered).detach().cpu().numpy())
        if it % 200000 == 0 or it == n_iter-1:
            plt.figure()
            plt.plot(loss_log)
            plt.figure()
            plt.plot(pts_dist_log, label="pts_dist")
            plt.plot(cam_dist_log, label="cam_dist")
            plt.legend()
            plot_camera_scene(cameras_absolute, cameras_absolute_gt,
                              points_absolute, points_gt,
                              'iteration=%3d; points_distance=%1.3e' % (it, points_3d_distance))

    print('Optimization finished.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
