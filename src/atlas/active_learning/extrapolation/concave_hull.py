"""Utilities to compute the concave hull for a point cloud of atomic descriptors."""

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull, Delaunay, KDTree
from shapely.affinity import scale
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point, Polygon
from shapely.ops import polygonize, unary_union

from atlas.core import code_utils as atl_cut

try:
    import numba

    njit = numba.njit(cache=True, fastmath=True)
except ImportError:  # fall back gracefully

    def njit(x=None, **kw):  # decorator that returns the function unchanged
        return x if x is not None else (lambda f: f)


def check_traj_in_domain(
    concave_hull: np.ndarray,
    descriptor_dict: dict,
    hull_scale_factor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list | np.ndarray | None]:
    """
    Check if the generated descriptors are inside the precomputed concave hull.

    Parameters
    ----------
    concave_hull : np.ndarray
        Concave hull of the latent space for the database, corresponding
        to its convex or concave hull.
    descriptor_dict : dict
        Descriptor dictionary containing the descriptors for each frame.
        The structure is as follows:
        ```python
        {
            uuid: {
                'latent_space': np.ndarray,
                'descriptors': np.ndarray,
                'is_extrapolating': np.ndarray,
            }
        }
        ```
    hull_scale_factor : float, optional
        Tolerance percentage to enlarge the concave hull.
        For example, 0.1 adds 10% tolerance. Default is 0.0.

    Returns
    -------
    np.ndarray
        Array containing the descriptors that are inside the concave hull.
    np.ndarray
        Array containing the descriptors that are outside the concave hull.
    np.ndarray
        Array containing boolean values showing if the frame is inside the concave
    """
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    # If the concave hull has multiple parts, create a MultiPolygon
    # from all the parts.
    if isinstance(concave_hull, list) and len(concave_hull) == 1:
        polygon = Polygon(concave_hull[0])
    elif isinstance(concave_hull, np.ndarray) and concave_hull.ndim == 2:
        polygon = Polygon(concave_hull)
    else:
        polygons = []
        for part in concave_hull:
            if len(part) >= 3:
                polygons.append(Polygon(part))
        polygon = MultiPolygon(polygons)

    # Scale the polygon if tolerance is provided
    scaled_hull_coords = None
    if hull_scale_factor > 0:
        scaling_factor = 1.0 + hull_scale_factor
        atl_cut.custom_print(
            f'Scaling concave hull by a factor of {scaling_factor} '
            'to apply tolerance...',
            'info',
        )
        polygon = scale(
            polygon, xfact=scaling_factor, yfact=scaling_factor, origin='center'
        )

        if isinstance(polygon, Polygon):
            scaled_hull_coords = np.array(polygon.exterior.coords)
        elif isinstance(polygon, MultiPolygon):
            scaled_hull_coords = [np.array(p.exterior.coords) for p in polygon.geoms]
    else:
        atl_cut.custom_print('No tolerance applied to concave hull.', 'warn')

    for c_uuid in descriptor_dict:
        descriptors = descriptor_dict[c_uuid]['latent_space']

        for _, frame_desc in enumerate(descriptors):
            c_all_p = []

            # In some cases, the frame_desc might be a 1D array instead of 2D
            # (e.g., only one descriptor).
            # We need to handle this case by reshaping it to (1, -1)
            # so that we can iterate over it consistently.
            if isinstance(frame_desc, np.ndarray) and frame_desc.ndim == 1:
                frame_desc = frame_desc.reshape(1, -1)

            for point in frame_desc:
                p = Point(point)
                if polygon.contains(p):
                    point_inside.append(point)
                    c_all_p.append(True)
                else:
                    point_outside.append(point)
                    c_all_p.append(False)

            all_points_in_out.append(np.array(c_all_p))

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out, scaled_hull_coords


def _edges_from_triangles(simplices):
    """
    Return the boundary edges from a set of triangles using vectorized NumPy operations.

    Parameters
    ----------
    simplices : (N, 3) int array
        Indices of points forming the triangles.

    Returns
    -------
    (K, 2) int array
        Array of boundary edges (indices).
    """
    # 1. Create all edges from the triangles (3 edges per triangle)
    # Shape: (3 * N, 2)
    edges = np.concatenate(
        [simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]]
    )

    # 2. Sort indices within each edge so (i, j) and (j, i) become identical
    # This allows us to count shared edges regardless of orientation
    edges.sort(axis=1)

    # 3. Find unique edges and how many times they appear
    # In a valid triangulation, internal edges appear exactly twice.
    # Boundary edges appear exactly once.
    unique_edges, counts = np.unique(edges, return_counts=True, axis=0)

    # 4. Keep only edges that appear once
    boundary_edges = unique_edges[counts == 1]

    return boundary_edges


def alpha_shape(points, alpha: float, only_outer: bool = True):
    """
    Compute the 2-D alpha-shape (concave hull) of a set of points.

    Parameters
    ----------
    points : (N, 2) array-like
        Input coordinates.
    alpha : float
        Inverse length scale.  Smaller alpha -> coarser (more concave) hull.
        A good starting point is alpha ~ 1 / (average edge length).
    only_outer : bool, default True
        If True, return only the outer boundary.  If False, keep holes.

    Returns
    -------
    shapely.geometry.Polygon | MultiPolygon
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)

    # Add small amount of noise to break perfect geometric alignments
    # 1e-9 should be small enough not to affect the data's meaning,
    # but large enough for Qhull to process it.
    # This should address the error from scipy.spatial._qhull.QhullError:
    # QH6019 qhull input error (qh_scalelast): can not scale last coordinate to...
    pts = pts + np.random.normal(0, 1e-9, pts.shape)

    # Some degenerate cases
    if pts.shape[0] <= 2:
        return LineString(pts)
    elif pts.shape[0] < 4:
        return Polygon(pts).convex_hull
    elif pts.shape[0] == 4:
        return Polygon(pts)

    tri = Delaunay(pts)
    simplices = tri.simplices  # (M, 3) int array

    # circum-radii of all Delaunay triangles
    pa, pb, pc = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]
    a = np.linalg.norm(pb - pc, axis=1)
    b = np.linalg.norm(pc - pa, axis=1)
    c = np.linalg.norm(pa - pb, axis=1)
    s = (a + b + c) * 0.5
    area = np.sqrt(np.clip(s * (s - a) * (s - b) * (s - c), 0.0, None))

    # Ignore division by zero and invalid operations
    with np.errstate(divide='ignore', invalid='ignore'):
        # R = abc / 4A
        circum_r = a * b * c / (4.0 * area)

    # keep triangles whose circum-radius satisfies R < 1/alpha
    keep = circum_r < 1.0 / alpha
    kept_simplices = simplices[keep]

    # boundary edges
    boundary_edges = _edges_from_triangles(kept_simplices)

    # build shapely geometry
    # boundary_edges is now a (K, 2) numpy array, so we can iterate directly
    segments = [LineString(pts[edge]) for edge in boundary_edges]
    mgeom = unary_union(segments)

    # build shapely geometry
    # segments = [LineString([pts[i], pts[j]]) for (i, j) in boundary_edges]
    # mgeom = unary_union(segments)

    polygons = list(polygonize(mgeom))

    if not polygons:
        return MultiPolygon()  # empty shape

    if only_outer:
        # Return the polygon with the largest area (outer shell)
        return max(polygons, key=lambda p: p.area)

    return MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]


def get_concave_hull_python(
    latent_space: np.ndarray,
    target_alpha_range: tuple[float, float] = (3.0, 8.0),
    default_alpha_if_issues: float = 5,
    nn_dist_scale_factor: float = 1.5,
    frac_points_allowed_out: float = 0.002,
    n_attempts: int = 20,
    decrease_factor_multiplier: float = 0.95,
    use_alpha: float | None = None,
) -> (np.ndarray, float):
    """
    Compute the concave hull of a set of points using the alpha-shape algorithm.

    Parameters
    ----------
    latent_space : np.ndarray
        The input points (N, 2) for which to compute the concave hull.
    target_alpha_range : tuple[float, float], optional
        The desired (min_alpha, max_alpha) range for the alpha parameter.
        Alpha will be clipped to this range. Defaults to (3.0, 8.0).
    default_alpha_if_issues : float, optional
        Default alpha value to use if nearest neighbor distance calculation
        is not possible (e.g., too few points) or other issues arise.
        Defaults to 5 (midpoint of common 3-8 range).
    nn_dist_scale_factor : float, optional
        Scaling factor for the alpha candidate calculation:
        `alpha_candidate = nn_dist_scale_factor / mean_nn_dist`.
        Defaults to 1.5.
    frac_points_allowed_out: float, optional
        The maximum fraction of points allowed to be outside the concave hull.
        If the fraction of points outside the hull exceeds this value,
        alpha will be decreased iteratively until the condition is met or
        alpha reaches zero. Defaults to 0.002 (0.2%).
    n_attempts : int, optional
        Number of attempts to compute the concave hull by adjusting alpha.
    use_alpha : float, optional
        If provided, this alpha value will be used directly to compute
        the concave hull without optimization. Defaults to None.

    Returns
    -------
    np.ndarray
        The coordinates of the vertices of the concave hull (N, 2).
        Returns an empty array np.empty((0,2)) if a hull cannot be formed.
    float
        The alpha value used to compute the concave hull.
    """
    total_num_points = latent_space.shape[0]

    if use_alpha is not None:
        atl_cut.custom_print(f'Using user-defined alpha: {use_alpha}', 'info')
        # Get alpha shape using the user-defined alpha
        shape = alpha_shape(latent_space, alpha=use_alpha, only_outer=True)

        exterior_xy = shape.exterior.coords.xy
        alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)

        # Check status of the points
        _, point_outside, _ = check_atom_in_domain(
            concave_hull=alpha_shape_arr, descriptors=latent_space
        )
        points_to_check = point_outside.shape[0]

        # Compute fraction of points outside the hull
        frac_outside = points_to_check / total_num_points

        atl_cut.custom_print(
            f'Current fraction of points outside hull is {frac_outside:.4f}.',
            'info',
        )

        return alpha_shape_arr, use_alpha

    else:
        alpha = default_alpha_if_issues

        try:
            tree = KDTree(latent_space)

            # k=2 for 1st nearest neighbor (k=1 is the point itself)
            distances, _ = tree.query(latent_space, k=2, workers=-1)
            nn_distances = distances[:, 1]

            # Filter out zero distances if any point is duplicated or coincident
            # Use a small epsilon for floating point comparisons
            valid_nn_distances = nn_distances[nn_distances > 1e-9]

            # If all points are coincident or extremely close
            if valid_nn_distances.size == 0:
                atl_cut.custom_print(
                    'All points are likely duplicates or extremely close. '
                    'Using max alpha from target range.',
                    'warn',
                )
                # Use max alpha, tends to shrink
                alpha = target_alpha_range[1]

            else:
                mean_nn_dist = np.mean(valid_nn_distances)

                # Effectively zero
                if mean_nn_dist < 1e-7:
                    # This case implies extremely dense points.
                    # A very large alpha candidate will be clipped to max of range.
                    alpha_candidate = np.inf
                    atl_cut.custom_print(
                        'Mean nearest neighbor distance is very small '
                        f'({mean_nn_dist:.2e}). Alpha candidate set to infinity'
                        ' before clipping.',
                        'debug',
                    )
                else:
                    alpha_candidate = nn_dist_scale_factor / mean_nn_dist
                    atl_cut.custom_print(
                        f'Mean NN dist: {mean_nn_dist:.4f}, '
                        f'NN Distance Scale Factor: {nn_dist_scale_factor:.1f}, '
                        f'Alpha candidate (factor/mean_nn_dist): {alpha_candidate:.4f}',
                        'debug',
                    )

                # Limiting alpha to the target range
                alpha = np.clip(
                    alpha_candidate, target_alpha_range[0], target_alpha_range[1]
                )
                atl_cut.custom_print(
                    f'Calculated alpha: {alpha:.4f}'
                    f' (clipped to range {target_alpha_range})',
                    'info',
                )

        except Exception as e:
            atl_cut.custom_print(
                f'Error during KDTree query or mean_nn_dist calculation: {e}. '
                f'Using default alpha: {default_alpha_if_issues}',
                'warn',
            )
            alpha = default_alpha_if_issues

        frac_outside = 1.0
        change_factor = alpha * 0.1
        min_change_factor = change_factor * 0.01
        num_attempts = 0
        alpha_lower_bound = 1.0

        while num_attempts <= n_attempts:
            print()
            atl_cut.custom_print(
                f'Computing alpha shape with alpha: {alpha:.4f}', 'info'
            )

            # Get alpha shape using the determined alpha
            shape = alpha_shape(latent_space, alpha=alpha, only_outer=True)

            if shape.is_empty:
                atl_cut.custom_print(
                    f'Alpha shape with alpha={alpha:.4f} resulted in an empty geometry.'
                    ' This can happen if alpha is too large (too restrictive)'
                    ' for the point set. Attempting to return convex hull'
                    ' as a fallback.',
                    'warn',
                )
                # Fallback to convex hull if alpha shape is empty
                if total_num_points >= 3:
                    try:
                        hull = ConvexHull(latent_space)
                        return latent_space[hull.vertices], 0
                    except Exception as e_cvx:  # pragma: no cover
                        atl_cut.custom_print(
                            f'Convex hull fallback also failed: {e_cvx}', 'error'
                        )
                        return np.empty((0, 2))
                else:
                    # Should have been caught earlier, but as a safeguard
                    return (
                        np.array(latent_space)
                        if total_num_points > 0
                        else np.empty((0, 2))
                    )

            if hasattr(shape.exterior, 'coords'):
                exterior_xy = shape.exterior.coords.xy
                alpha_shape_arr = np.stack((exterior_xy[0], exterior_xy[1]), axis=1)
            else:
                atl_cut.custom_print(
                    f'Alpha shape (alpha={alpha:.4f}) did not return a simple polygon '
                    'with an exterior. '
                    f'Shape type: {type(shape)}. This is unexpected '
                    'with only_outer=True.  Attempting to return convex'
                    ' hull as a fallback.',
                    'error',
                )
                if total_num_points >= 3:
                    try:
                        hull = ConvexHull(latent_space)
                        return latent_space[hull.vertices]
                    except Exception as e_cvx2:  # pragma: no cover
                        atl_cut.custom_print(
                            f'Convex hull fallback also failed: {e_cvx2}', 'error'
                        )
                        return np.empty((0, 2))
                else:
                    return (
                        np.array(latent_space)
                        if total_num_points > 0
                        else np.empty((0, 2))
                    )

            # Check status of the points
            _, point_outside, _ = check_atom_in_domain(
                concave_hull=alpha_shape_arr, descriptors=latent_space
            )
            points_to_check = point_outside.shape[0]

            # Compute fraction of points outside the hull
            prev_frac_outside = frac_outside
            frac_outside = points_to_check / total_num_points
            prev_distance_to_target = abs(prev_frac_outside - frac_points_allowed_out)
            distance_to_target = abs(frac_outside - frac_points_allowed_out)

            # Decrease alpha to make hull less strict
            last_alpha = alpha

            # Change the sign of the change factor if we overshot
            if distance_to_target <= prev_distance_to_target:
                change_factor = abs(change_factor)
            else:
                change_factor = -1 * abs(change_factor)

            # alpha -= change_factor
            alpha -= change_factor

            # Decrease the factor for more gradual changes next iteration
            change_factor = np.max(
                (change_factor * decrease_factor_multiplier, min_change_factor)
            )

            if alpha <= alpha_lower_bound:
                atl_cut.custom_print(
                    f'Alpha has reached the minimum threshold of {alpha_lower_bound}. '
                    'Stopping adjustments.',
                    'warn',
                )
                break

            if frac_outside >= frac_points_allowed_out:
                atl_cut.custom_print(
                    (
                        f'Current fraction of points outside hull is'
                        f' {frac_outside:.4e}, above the allowed'
                        f' {frac_points_allowed_out:.4e} threshold.'
                    ),
                    'warn',
                )
                atl_cut.custom_print(f'Area of concave hull: {shape.area:.4e}', 'info')
                atl_cut.custom_print(f'Decreasing alpha to: {alpha:.4f}', 'info')
                last_alpha = alpha
                num_attempts += 1
            else:
                atl_cut.custom_print(
                    (
                        f'Current fraction of points outside hull'
                        f' is {frac_outside:.4e}, '
                        f' below the allowed {frac_points_allowed_out:.4e} threshold.'
                    ),
                    'info',
                )
                break

        # After optimizing alpha, print final status.
        print()
        atl_cut.custom_print(
            (
                f'Current fraction of points outside hull is {frac_outside:.4f}, '
                f'after {num_attempts} attempts.'
            ),
            'info',
        )
        atl_cut.custom_print(f'Final alpha: {last_alpha}', 'done')
        print()

    return alpha_shape_arr, last_alpha


def get_optimized_concave_hull(
    latent_space: np.ndarray,
    target_alpha_range: tuple[float, float] = (0.0, 10.0),
    frac_points_allowed_out: float = 0.02,
) -> tuple[np.ndarray, float]:
    total_points = len(latent_space)

    # Heuristic: Linearity Check (PCA)
    # In simple and small datasets, the points may be collinear.
    # In that case, the concave hull is ill-defined.
    # Hence, we do a linearity check, and if not passed, we
    # return the convex hull instead.

    # Center the data
    centered_data = latent_space - np.mean(latent_space, axis=0)

    # Compute covariance matrix and eigenvalues
    # This measures the spread of data in orthogonal directions
    cov = np.cov(centered_data.T)
    eigenvalues = np.linalg.eigvalsh(cov)

    # Sort: [smaller_spread, larger_spread]
    eigenvalues = np.sort(eigenvalues)

    # Calculate aspect ratio
    # If the smaller eigenvalue is near zero, or the ratio is extreme, it's a line.
    # eps to avoid division by zero
    val_min = max(eigenvalues[0], 1e-12)
    val_max = eigenvalues[-1]
    aspect_ratio = val_max / val_min

    # Threshold: if the collection of descriptors is 500x longer than it is wide,
    # treat as a line.
    if aspect_ratio > 500.0 or total_points <= 2:
        atl_cut.custom_print(
            f'Linear Geometry detected (Ratio {aspect_ratio:.1f}). '
            'Returning Convex Hull.',
            'warning',
        )
        hull = MultiPoint(latent_space).convex_hull

        # Handle the edge case of perfectly collinear points (area = 0)
        if hull.area < 1e-9:
            # Buffer it slightly so it becomes a valid polygon for later checks
            hull = hull.buffer(0.01 * val_max)

        return np.array(hull.exterior.coords), 0.0

    # Normal case: Non-linear geometry
    # This runs an optimizer to find the best alpha value.

    # Normalize data to [0, 1]
    # This allows to use consistent alpha ranges across different datasets
    # by removing scale effects.
    min_vals = latent_space.min(axis=0)
    max_vals = latent_space.max(axis=0)
    span = max_vals - min_vals

    # Avoid division by zero
    span[span == 0] = 1.0
    norm_latent_space = (latent_space - min_vals) / span

    # Define the objective function
    def objective(alpha, norm_latent_space):
        # Handle edge case where alpha is too small (convexhull-ish)
        if alpha <= 0:
            return 1e9 - (alpha * 1000)

        # Compute the alpha shape
        try:
            # only_outer=True to get the polygon
            shape = alpha_shape(norm_latent_space, alpha=alpha, only_outer=True)
        except Exception:
            # Penalty for failed shape generation, guided by alpha.
            # getting stuck at large alpha values that cause failures.
            return 1e9 + (alpha * 1000)

        # If shape is empty or invalid geometry
        # Penalizing by alpha to avoid getting stuck at
        # large alpha values that cause empty shapes.
        if shape.is_empty or not hasattr(shape, 'area'):
            cost = 1e9 + (alpha * 1000)
            atl_cut.custom_print(
                'Shape is empty for '
                f'Alpha: {alpha:.4f}, '
                f'Cost: {1e9 + (alpha * 1000):.5e}',
                'warning',
            )
            return cost

        # Constraint check
        # Check if the random points are inside the concave hull.
        shape = scale(shape, xfact=1.015, yfact=1.015, origin='centroid')
        point_inside, point_outside, all_points = check_atom_in_domain(
            shape, norm_latent_space
        )
        frac_outside = point_outside.shape[0] / total_points

        # Cost calculation
        if frac_outside > frac_points_allowed_out:
            # Penalizing going outside the allowed fraction.
            # We add a massive penalty so the optimizer leaves this region immediately.
            # We add frac_outside to give it a 'gradient' to follow back to safety.
            cost = 1e6 + (frac_outside * 1000)
        else:
            # We want to minimize area.

            # However, we must ensure the optimizer doesn't pick a massive alpha
            # that results in a tiny, fragmented hull just to get low area

            # We penalize area by alpha^2 to ensure that the optimizer doesn't pick
            # a very small alpha every time.

            # We add a small epsilon to alpha^2 to avoid division by zero
            # or the cost blowing up with very small alpha.
            cost = shape.area / (alpha**2 + 1e-4)

        atl_cut.custom_print(
            f'Alpha: {alpha:.4f}, '
            f'Points inside: {point_inside.shape[0]}, '
            f'Points outside: {point_outside.shape[0]}, '
            f'Points total: {total_points}, '
            f'Fraction outside: {frac_outside:.4f}, '
            f'Normalized Area: {shape.area:.4e}, '
            f'Cost: {cost:.5e}',
            'info',
        )
        return cost

    # Enforce bounds
    lower_bound = max(0.001, target_alpha_range[0])
    upper_bound = min(350.0, target_alpha_range[1])

    # Run the optimizer
    # Method 'bounded' is best when we know the min/max alpha
    # Tolerance: stop when alpha changes by less than xatol
    result = minimize_scalar(
        objective,
        bounds=(lower_bound, upper_bound),
        method='bounded',
        options={'xatol': 0.00125},
        args=(norm_latent_space,),
    )
    best_alpha = result.x
    # final_shape = alpha_shape(latent_space, alpha=best_alpha, only_outer=True)

    # Denormalize the result
    final_shape_norm = alpha_shape(norm_latent_space, alpha=best_alpha, only_outer=True)

    # Extract normalized coords
    if hasattr(final_shape_norm, 'exterior') and hasattr(
        final_shape_norm.exterior, 'coords'
    ):
        exterior_norm = np.array(final_shape_norm.exterior.coords)
    elif isinstance(final_shape_norm, MultiPolygon) and final_shape_norm.is_empty:
        alpha_shape_arr = np.empty((0, 2))
        atl_cut.custom_print(
            f'Alpha={best_alpha:.4f} resulted in empty geometry.',
            'warning',
        )

    else:
        exterior_norm = np.array(final_shape_norm.convex_hull.exterior.coords)

    if not final_shape_norm.is_empty:
        # Rescale back to original units
        alpha_shape_arr = exterior_norm * span + min_vals

        atl_cut.custom_print(
            f'Optimized alpha: {best_alpha:.4f} with non-normalized area: '
            f'{Polygon(alpha_shape_arr).area:.4e}. '
            f'Number of points inside: {total_points}.',
            'done',
        )

    return alpha_shape_arr, best_alpha


def check_atom_in_domain(
    concave_hull: np.ndarray,
    descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Check if the generated descriptors are inside the precomputed concave hull.

    Parameters
    ----------
    concave_hull : np.ndarray
        Concave hull of the latent space for the database, corresponding
        to its convex or concave hull.
    descriptors : np.ndarray
        The descriptors to check, typically of shape (N, 2).

    Returns
    -------
    np.ndarray
        Array containing the descriptors that are inside the concave hull.
    np.ndarray
        Array containing the descriptors that are outside the concave hull.
    np.ndarray
        Array containing boolean values showing if the frame is inside the concave hull.
    """
    point_inside = []
    point_outside = []
    all_points_in_out = []

    # Check if the random points are inside the bounds of the
    # concave hull by checking if the points are inside the
    # polygon formed by the concave hull.
    polygon = Polygon(concave_hull)
    for point in descriptors:
        p = Point(point)
        if polygon.contains(p):
            point_inside.append(point)
            all_points_in_out.append(True)
        else:
            point_outside.append(point)
            all_points_in_out.append(False)

    point_inside = np.array(point_inside)
    point_outside = np.array(point_outside)
    all_points_in_out = np.array(all_points_in_out)
    return point_inside, point_outside, all_points_in_out


def plot_concave_hull(
    concave_hull: np.ndarray,
    latent_space: np.ndarray = None,
    point_inside: np.ndarray = None,
    point_outside: np.ndarray = None,
    scaled_hull: np.ndarray | list[np.ndarray] = None,
    filename: str = None,
    alpha: float = None,
    plot_density: bool = False,
    title: str = 'Concave Hull',
):
    """
    Generate a plot for the concave hull area in 2D space, including in and out
    of domain points if provided.

    Parameters
    ----------
    concave_hull : np.ndarray
        The coordinates of the concave hull vertices (N, 2).
    latent_space : np.ndarray, optional
        The full set of points in the latent space (N, 2).
    point_inside : np.ndarray, optional
        Points inside the concave hull (M, 2).
    point_outside : np.ndarray, optional
        Points outside the concave hull (K, 2).
    scaled_hull : np.ndarray | list[np.ndarray], optional
        The coordinates of the scaled concave hull vertices (N, 2).
    filename : str, optional
        The filename to save the plot. Defaults to 'concave_hull.png'.
    alpha : float, optional
        If provided, a single alpha value will be used used to compute the concave hull,
        instead of attempting to optimize it.
    plot_density : bool, optional
        Whether to color points by density. Defaults to False.
    title : str, optional
        The title of the plot. Defaults to 'Concave Hull'.
    """
    if not filename:
        filename = 'concave_hull.png'

    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate limits based on all available data
    all_points = []
    for arr in (latent_space, point_inside, point_outside):
        if arr is not None and arr.size > 0:
            all_points.append(np.atleast_2d(arr))
    if isinstance(concave_hull, list):
        for hull in concave_hull:
            all_points.append(np.atleast_2d(np.asarray(hull)))
    elif concave_hull is not None:
        all_points.append(np.atleast_2d(np.asarray(concave_hull)))

    if all_points:
        all_points_stacked = np.vstack(all_points)
        x_min, y_min = np.min(all_points_stacked, axis=0)
        x_max, y_max = np.max(all_points_stacked, axis=0)

        # Add some padding
        padding = 0.05
        w = x_max - x_min
        h = y_max - y_min
        x_min -= w * padding
        x_max += w * padding
        y_min -= h * padding
        y_max += h * padding

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    # Plotting latent space points using Datashader
    if latent_space is not None and latent_space.size != 0:
        df = pd.DataFrame(latent_space, columns=['x', 'y'])

        # Use the plot limits for the canvas
        x_min_lim, x_max_lim = ax.get_xlim()
        y_min_lim, y_max_lim = ax.get_ylim()

        # Create datashader canvas
        cvs = ds.Canvas(
            plot_width=800 * 2,
            plot_height=800 * 2,
            x_range=(x_min_lim, x_max_lim),
            y_range=(y_min_lim, y_max_lim),
        )
        agg = cvs.points(df, 'x', 'y')

        # Shade the points - using a blue colormap for density
        img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')

        # Convert to PIL image and display with imshow
        img_pil = img.to_pil()
        ax.imshow(
            img_pil,
            extent=[x_min_lim, x_max_lim, y_min_lim, y_max_lim],
            origin='upper',
            aspect='auto',
            zorder=0,
        )

        # Add dummy legend entry
        ax.scatter([], [], c='darkblue', label='Descriptor in database', s=10)

    if point_inside is not None and point_inside.size != 0:
        ax.scatter(
            point_inside[:, 0],
            point_inside[:, 1],
            marker='s',
            label='Structure in domain',
            color='#8ec07c',
            s=20,
            linewidths=0.5,
            edgecolors='#282828',
            zorder=3,
        )
    if point_outside is not None and point_outside.size != 0:
        ax.scatter(
            point_outside[:, 0],
            point_outside[:, 1],
            marker='s',
            label='Structure out of domain',
            color='#fb4934',
            s=20,
            linewidths=0.5,
            edgecolors='#282828',
            zorder=3,
        )

    # Plot the concave hull boundary
    if isinstance(concave_hull, list):
        for i, hull in enumerate(concave_hull):
            ax.plot(
                hull[:, 0],
                hull[:, 1],
                '-',
                color='#cc241d',
                lw=2,
                label='Concave hull' if i == 0 else None,
                zorder=2,
            )
        polygons = [Polygon(hull) for hull in concave_hull]
        # Valid polygons need at least 3 points and must not be empty
        polygons = [p for p in polygons if p.is_valid and not p.is_empty]

        if not polygons:
            hull_area = 0.0
        elif len(polygons) == 1:
            hull_area = polygons[0].area
        else:
            hull_area = MultiPolygon(polygons).area
    else:
        ax.plot(
            concave_hull[:, 0],
            concave_hull[:, 1],
            '-',
            color='#cc241d',
            lw=2,
            label='Concave hull',
            zorder=2,
        )

        polygon = Polygon(concave_hull)
        hull_area = polygon.area

    # Plot scaled hull if provided
    if scaled_hull is not None:
        if isinstance(scaled_hull, list):
            for i, hull in enumerate(scaled_hull):
                ax.plot(
                    hull[:, 0],
                    hull[:, 1],
                    '--',
                    color='#d65d0e',
                    lw=2,
                    label='Scaled hull' if i == 0 else None,
                    zorder=2,
                )
        else:
            ax.plot(
                scaled_hull[:, 0],
                scaled_hull[:, 1],
                '--',
                color='#d65d0e',
                lw=2,
                label='Scaled hull',
                zorder=2,
            )

    scaled_hull_area = None
    if scaled_hull is not None:
        if isinstance(scaled_hull, list):
            polygons_scaled = [Polygon(h) for h in scaled_hull]
            # Valid polygons need at least 3 points and must not be empty
            polygons_scaled = [
                p for p in polygons_scaled if p.is_valid and not p.is_empty
            ]

            if not polygons_scaled:
                scaled_hull_area = 0.0
            elif len(polygons_scaled) == 1:
                scaled_hull_area = polygons_scaled[0].area
            else:
                scaled_hull_area = MultiPolygon(polygons_scaled).area
        else:
            scaled_hull_area = Polygon(scaled_hull).area

    if alpha is None:
        alpha = 'unknown'
    if isinstance(alpha, (float, int)):
        alpha = f'{alpha:.2f}'

    text_str = f'Alpha: {alpha}\nHull area: {hull_area:.2e}'

    if scaled_hull_area is not None:
        text_str += f'\nScaled area: {scaled_hull_area:.2e}'

    if point_inside is not None or point_outside is not None:
        n_in = point_inside.shape[0] if point_inside is not None else 0
        n_out = point_outside.shape[0] if point_outside is not None else 0
        total = n_in + n_out
        if total > 0:
            frac_in = n_in / total
            frac_out = n_out / total
            text_str += f'\nIn domain: {frac_in:.1%}'
            text_str += f'\nOut domain: {frac_out:.1%}'

    # Write area and alpha in a text box
    plt.text(
        0.05,
        0.95,
        text_str,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        zorder=4,
    )

    plt.title(title, fontsize=16)
    plt.xlabel('Embedded dimension 1', fontsize=14)
    plt.ylabel('Embedded dimension 2', fontsize=14)
    plt.legend(fontsize=12)

    # Save as PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Save as SVG
    if hasattr(filename, 'with_suffix'):
        filename_svg = filename.with_suffix('.svg')
    else:
        filename_svg = f'{filename}.svg'
    plt.savefig(filename_svg, dpi=300, bbox_inches='tight')

    plt.close(fig)


if __name__ == '__main__':
    plot_hull = True
    latent_space_file = 'latent_space.npy'
    descriptors_file = 'descriptors.npz'

    # Set the random seed for reproducibility.
    rng = np.random.default_rng(seed=420)

    # Gather the latent space from the autoencoder.
    latent_space = np.load(latent_space_file)

    # Get descriptors array
    descriptors = np.load(descriptors_file)

    # Compute the concave hull.
    concave_hull, alpha = get_concave_hull_python(latent_space)

    # Check if the random points are inside the concave hull.
    point_inside, point_outside, all_points = check_atom_in_domain(
        concave_hull, descriptors
    )

    np.save('point_inside.npy', point_inside)
    np.save('point_outside.npy', point_outside)
    np.save('all_points.npy', all_points)

    if plot_hull:
        plot_concave_hull(
            concave_hull=concave_hull,
            point_inside=point_inside,
            point_outside=point_outside,
            latent_space=latent_space,
            filename='/tmp/concave_hull.png',
        )
