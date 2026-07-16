"""Utility functions for QuadTree operations."""

import pathlib as pl
from dataclasses import dataclass

import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon


@dataclass
class Rectangle:
    """
    Object representing a rectangle for a QuadTree.

    Axis-aligned rectangle represented by its center (x, y)
    and half-width (w) and half-height (h).
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def xmin(self) -> float:
        return self.x - self.w

    @property
    def xmax(self) -> float:
        return self.x + self.w

    @property
    def ymin(self) -> float:
        return self.y - self.h

    @property
    def ymax(self) -> float:
        return self.y + self.h

    def contains(self, point) -> bool:
        # Refactored to use the properties for consistency
        return self.xmin <= point.x < self.xmax and self.ymin <= point.y < self.ymax

    def get_area(self) -> float:
        return (self.w * 2) * (self.h * 2)


class QuadTree:
    """A QuadTree for spatial partitioning of 2D points.

    The QuadTree subdivides space into quadrants to efficiently manage and query points.
    The reasoning behind its use is to identify dense regions in a 2D space by
    recursively subdividing areas until a certain density criterion is met, which
    allows to select regions for alpha-shape computation in multi-resolution datasets.
    """

    def __init__(
        self,
        boundary: Rectangle,
        capacity: int = 4,
        initial_capacity_fraction: float | None = None,
        initial_data_amount: int = 0,
        data_range_x: float | None = None,
        data_range_y: float | None = None,
    ):
        self.boundary = boundary
        self.capacity = capacity
        self.initial_capacity_fraction: float = initial_capacity_fraction
        self.initial_data_amount: int = initial_data_amount
        self.points: list[Point] = []
        self.divided: bool = False
        self.northwest: QuadTree | None = None
        self.northeast: QuadTree | None = None
        self.southwest: QuadTree | None = None
        self.southeast: QuadTree | None = None

        self.data_range_x = data_range_x
        self.data_range_y = data_range_y

    def subdivide(self) -> None:
        x, y, w, h = self.boundary.x, self.boundary.y, self.boundary.w, self.boundary.h

        # Define new capacity for child nodes. Check if initial_capacity_fraction is set
        # otherwise keep capacity constant.
        if self.initial_capacity_fraction and self.initial_capacity_fraction > 0:
            calculated_cap = int(len(self.points) * self.initial_capacity_fraction)
            # Ensure we don't drop below 5 to prevent infinite loops
            new_capacity = max(5, calculated_cap)
        else:
            # Standard QuadTree behavior
            new_capacity = self.capacity

        self.northeast = QuadTree(
            Rectangle(x + w / 2, y - h / 2, w / 2, h / 2),
            capacity=new_capacity,
            initial_capacity_fraction=self.initial_capacity_fraction,
        )
        self.northwest = QuadTree(
            Rectangle(x - w / 2, y - h / 2, w / 2, h / 2),
            capacity=new_capacity,
            initial_capacity_fraction=self.initial_capacity_fraction,
        )
        self.southeast = QuadTree(
            Rectangle(x + w / 2, y + h / 2, w / 2, h / 2),
            capacity=new_capacity,
            initial_capacity_fraction=self.initial_capacity_fraction,
        )
        self.southwest = QuadTree(
            Rectangle(x - w / 2, y + h / 2, w / 2, h / 2),
            capacity=new_capacity,
            initial_capacity_fraction=self.initial_capacity_fraction,
        )

        # Redistribute existing points to children
        for point in self.points:
            # We don't check boundary again because we know they are in parent
            if self.northeast.insert(point):
                continue
            if self.northwest.insert(point):
                continue
            if self.southeast.insert(point):
                continue
            if self.southwest.insert(point):
                continue

        # Parent becomes purely structural since all points
        # are now in children
        self.points.clear()

        self.divided = True

    def insert(self, point: Point) -> bool:
        if not self.boundary.contains(point):
            return False

        # If it's a leaf node and has space, store the point
        if not self.divided:
            if len(self.points) < self.capacity:
                self.points.append(point)
                return True
            else:
                # If the box is full, we need to subdivide and
                # then try inserting into children
                self.subdivide()

        # We know children exist after subdivide
        if self.northeast.insert(point):
            return True
        if self.northwest.insert(point):
            return True
        if self.southeast.insert(point):
            return True
        if self.southwest.insert(point):  # noqa
            return True
        return False

    def find_dense_leaves_density(
        self, min_density_threshold: float, total_area: float
    ) -> list[Rectangle]:
        """
        Traverse the tree and return leaf nodes that meet a density requirement.
        Density = (number of points) / (area of node).
        """
        dense_regions = []

        if not self.divided:
            if len(self.points) > 0:
                # Calculate simple density: points / area
                node_area = self.boundary.get_area()
                density = len(self.points) / node_area

                if density >= min_density_threshold:
                    dense_regions.append(self.boundary)
            return dense_regions

        children = [self.northwest, self.northeast, self.southwest, self.southeast]
        for child in children:
            if child:
                dense_regions.extend(
                    child.find_dense_leaves_density(min_density_threshold, total_area)
                )

        return dense_regions

    def find_dense_leaves(self, max_width_threshold: float) -> list[Rectangle]:
        """
        Traverse the tree and return the boundaries of leaf nodes that are
        smaller than the threshold (indicating high density).

        Parameters
        ----------
        max_width_threshold : float
            The width (2 * w) below which a node is considered 'dense'.

        Returns
        -------
        list[Rectangle]
            A list of bounding boxes representing dense regions.
        """
        dense_regions = []

        # If this is a leaf node (not divided)
        if not self.divided:
            # Check if it actually contains data (ignore empty leaves)
            if len(self.points) > 0:
                # Check if the node is small enough to be considered a cluster
                node_width = self.boundary.w * 2
                if node_width <= max_width_threshold:
                    dense_regions.append(self.boundary)
            return dense_regions

        # If divided, recurse into children
        # (Using filter to ensure we only traverse existing children)
        children = [self.northwest, self.northeast, self.southwest, self.southeast]
        for child in children:
            if child:
                dense_regions.extend(child.find_dense_leaves(max_width_threshold))

        return dense_regions


def setup_quadtree(
    all_points, offset_frac: float = 0.1, data_frac_capacity: float = 0.015
):
    # Define boundary
    min_x = min(p.x for p in all_points)
    max_x = max(p.x for p in all_points)
    min_y = min(p.y for p in all_points)
    max_y = max(p.y for p in all_points)

    range_x = max_x - min_x
    range_y = max_y - min_y
    offset_frac = 0.1
    min_x -= range_x * offset_frac
    max_x += range_x * offset_frac
    min_y -= range_y * offset_frac
    max_y += range_y * offset_frac

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    width = (max_x - min_x) / 2
    height = (max_y - min_y) / 2
    boundary = Rectangle(cx, cy, width, height)

    # 2. Build QuadTree
    qt_capacity = max(1, int(len(all_points) * data_frac_capacity))
    qt = QuadTree(
        boundary,
        capacity=qt_capacity,
        initial_data_amount=len(all_points),
        data_range_x=range_x,
        data_range_y=range_y,
    )
    for p in all_points:
        qt.insert(p)

    return qt


def separate_clusters(dense_boxes: list[Rectangle]) -> list[list[Rectangle]]:
    """
    Groups dense boxes into connected clusters.
    Boxes are considered connected if they touch or overlap.
    """
    if not dense_boxes:
        return []

    # Sort boxes by their left edge x-coordinate
    # to allow pruning optimization
    boxes = sorted(dense_boxes, key=lambda b: b.x - b.w)
    n = len(boxes)

    adjacency = {i: [] for i in range(n)}

    # Use a small epsilon for float comparison stability
    eps = 1e-9

    # Build adjacency graph
    for i in range(n):
        b1 = boxes[i]
        b1_max_x = b1.x + b1.w

        for j in range(i + 1, n):
            b2 = boxes[j]
            b2_min_x = b2.x - b2.w

            # If b2 starts after b1 ends, no subsequent box (j+1, j+2...)
            # can possibly touch b1 since they are sorted by x.
            if b2_min_x > b1_max_x + eps:
                break

            x_dist = abs(b1.x - b2.x)
            y_dist = abs(b1.y - b2.y)

            w_sum = b1.w + b2.w
            h_sum = b1.h + b2.h

            # Check for intersection or touching
            if x_dist <= w_sum + eps and y_dist <= h_sum + eps:
                adjacency[i].append(j)
                adjacency[j].append(i)

    # Find connected components to identify clusters
    visited = set()
    clusters = []

    for i in range(n):
        if i not in visited:
            cluster = []
            stack = [i]
            visited.add(i)

            while stack:
                node_idx = stack.pop()
                cluster.append(boxes[node_idx])

                for neighbor in adjacency[node_idx]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            clusters.append(cluster)

    return clusters


def visualize_quadtree(
    qt: QuadTree,
    points: list[Point],
    clusters: list[list[Rectangle]],
    alpha_shapes: list[dict] | None = None,
    filename: str | pl.Path = 'quadtree_viz.png',
    frac_outside: float = 0.0,
    show: bool = False,
):
    """Visualizes the quadtree structure, data points, and clusters."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Set plot limits based on the root boundary
    x_min = qt.boundary.x - qt.boundary.w
    x_max = qt.boundary.x + qt.boundary.w
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(qt.boundary.y - qt.boundary.h, qt.boundary.y + qt.boundary.h)

    # Configure x-axis ticks
    steps = x_max - x_min
    interval = 1
    if steps > 100:
        interval = 20
    elif steps > 50:
        interval = 10
    elif steps > 20:
        interval = 5

    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))

    # Helper to draw quadtree rectangles recursively
    def draw_rects(node: QuadTree):
        # Draw current node boundary
        x, y = node.boundary.x - node.boundary.w, node.boundary.y - node.boundary.h
        w, h = node.boundary.w * 2, node.boundary.h * 2
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=1, edgecolor='lightgray', facecolor='none'
        )
        ax.add_patch(rect)

        if node.divided:
            if node.northwest:
                draw_rects(node.northwest)
            if node.northeast:
                draw_rects(node.northeast)
            if node.southwest:
                draw_rects(node.southwest)
            if node.southeast:
                draw_rects(node.southeast)

    # Draw the QuadTree Grid
    draw_rects(qt)

    # Draw Clusters (Highlighted with different colors)
    # Get a colormap to differentiate clusters
    cmap = plt.get_cmap('tab10')

    n_clusters = len(clusters)

    for i, cluster in enumerate(clusters):
        color = cmap(i % 10)
        for j, box in enumerate(cluster):
            x, y = box.x - box.w, box.y - box.h
            w, h = box.w * 2, box.h * 2

            # Label only the first box of the cluster for the legend
            if n_clusters > 5:
                label = f'{i + 1}' if j == 0 else None
            else:
                label = f'Cluster {i + 1}' if j == 0 else None

            rect = patches.Rectangle(
                (x, y),
                w,
                h,
                linewidth=2,
                edgecolor=color,
                facecolor=color,
                alpha=0.4,
                label=label,
            )
            ax.add_patch(rect)
            ax.plot(box.x, box.y, marker='x', color='black', markersize=5)

    df = pd.DataFrame([(p.x, p.y) for p in points], columns=['x', 'y'])

    # Scatter Plot Points (Datashader)
    if len(points) > int(5e4):
        # Use the plot limits for the canvas
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        # Create datashader canvas
        cvs = ds.Canvas(
            plot_width=800 * 2,
            plot_height=800 * 2,
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
        )
        agg = cvs.points(df, 'x', 'y')
        img = tf.shade(agg, cmap=['lightblue', 'darkblue'], how='log')
        img_pil = img.to_pil()
        ax.imshow(
            img_pil,
            extent=[x_min, x_max, y_min, y_max],
            origin='upper',
            aspect='auto',
            zorder=2,
        )
        ax.scatter([], [], c='darkblue', label='Data Points (Density)', s=10)
    else:
        ax.scatter(df['x'], df['y'], c='darkblue', label='Data Points (Density)', s=10)

    # Shade the points - using a blue colormap for density

    # Convert to PIL image and display with imshow

    # Add dummy legend entry

    # Draw Alpha Shapes
    if alpha_shapes:
        for item in alpha_shapes:
            shape = item['alpha_shape']

            if shape is None or shape.is_empty:
                continue

            # Handle different geometry types (Polygon or MultiPolygon)
            geoms = [shape] if isinstance(shape, Polygon) else []
            if isinstance(shape, MultiPolygon):
                geoms = list(shape.geoms)

            for geom in geoms:
                x, y = geom.exterior.xy
                # Only label the first alpha shape component for the legend
                lbl = (
                    'Alpha Shape'
                    if item['cluster_id'] == 1 and geom == geoms[0]
                    else None
                )
                ax.plot(x, y, color='#cc241d', linewidth=1, linestyle='--', label=lbl)

            if 'frac_inside_hull' in item:
                cx, cy = shape.centroid.x, shape.centroid.y
                alpha_val = item.get('used_alpha', 0.0)
                ax.text(
                    cx,
                    cy,
                    f'alpha = {alpha_val:.2f}\n'
                    f'In:{item["frac_inside_hull"] * 100:.1f}%',
                    color='#cc241d',
                    fontsize=9,
                    ha='center',
                    va='center',
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1),
                )

    ax.set_title('Latent-Space Boundary Visualization')
    ax.set_xlabel('Latent space x')
    ax.set_ylabel('Latent space y')

    if frac_outside > 0:
        ax.text(
            0.85,
            0.01,
            f'% points outside alpha shapes: {frac_outside * 100:.2f}%',
            transform=ax.transAxes,
            ha='center',
            fontsize=12,
            color='gray',
        )

    # Reorder legend to show clusters nicely
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))

    if n_clusters <= 15:
        ax.legend(by_label.values(), by_label.keys())

    plt.savefig(filename, dpi=600, bbox_inches='tight')
    print(f'Visualization saved to {filename}')
    if show:
        plt.show()
    plt.close()


def check_if_points_in_polygons(alpha_shapes: list, data: list[Point]):
    alpha_shapes_list = []
    for cluster in alpha_shapes:
        polygon = cluster.get('alpha_shape')
        alpha_shapes_list.append(polygon)

    points_inside = []
    points_outside = []
    for point in data:
        added_point = False
        for polygon in alpha_shapes_list:
            if polygon is not None and polygon.contains(point):
                points_inside.append(point)
                added_point = True
                break

        if not added_point:
            points_outside.append(point)

    frac_outside = len(points_outside) / (len(points_inside) + len(points_outside))
    return points_inside, points_outside, frac_outside
