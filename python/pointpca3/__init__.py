import numpy as np
from .pointpca3 import compute_pointpca3 as __pointpca3_internal


def __preprocess_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape != colors.shape:
        raise Exception("Points and colors must have the same shape.")
    is_normalized = colors.max() <= 1 and colors.min() >= 0
    is_floating = np.issubdtype(colors.dtype, np.floating)
    if is_normalized and is_floating:
        colors *= 255
    if points.dtype != np.double:
        points = points.astype(np.double)
    if colors.dtype != np.uint8:
        colors = colors.astype(np.uint8)
    return points, colors


def compute_pointpca3(
    points_reference: np.ndarray,
    colors_reference: np.ndarray,
    points_test: np.ndarray,
    colors_test: np.ndarray,
    search_size=81,
    max_workers=0,
    verbose=False,
) -> np.ndarray:
    """
    Compute PointPCA3 from reference and test point clouds.

    Parameters
    ----------
    points_reference : (M, 3) array_like
        Points from the reference point cloud.
    colors_reference : (M, 3) array_like
        Colors from the reference point cloud.
    points_test : (M, 3) array_like
        Points from the test point cloud.
    colors_test : (M, 3) array_like
        Colors from the test point cloud.
    search_size : int, optional
        The k-nearest-neighbors search size.
        Default is 81.
    max_workers : int, optional
        Sets the number of threads to be used in the thread-pool.
        If you specify a non-zero number of threads using this parameter, then the
        resulting thread-pools are guaranteed to start at most this number of threads.
        If max_workers is 0, then the number of logical CPUs will be used.
        Default is 0.
    verbose : bool, optional
        Whether to display verbose information or not.
        Default is False.

    Returns
    -------
    pointpca3 : (1, 40) np.ndarray
        The computed PointPCA3 features (predictors).

    Notes
    -----
    For the points arguments, any kind of dtype is accepted, but
    the array will eventually be converted to np.double.

    For the colors arguments, it is expected that the colors are
    on the [0, 1] range (np.double), or [0, 255]. Other ranges are not supported.
    Any dtype is accepted, but the array will eventually be converted
    to np.uint8.

    It is recommended to simply read the point cloud using open3d,
    and pass the points and colors parameters as np.ndarrays.

    Point clouds without colors currently are not supported.
    """
    points_a, colors_a = __preprocess_point_cloud(points_reference, colors_reference)
    points_b, colors_b = __preprocess_point_cloud(points_test, colors_test)
    predictors = __pointpca3_internal(
        points_a,
        colors_a,
        points_b,
        colors_b,
        search_size,
        max_workers,
        verbose,
    )
    return predictors
