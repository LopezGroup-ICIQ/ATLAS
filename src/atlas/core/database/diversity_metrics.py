"""Contains functions used to compute completeness metrics on the database."""

import pathlib as pl
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, normalize

from atlas.active_learning.active_learning_utils import (
    generate_descriptors,
    get_species_from_database,
)
from atlas.core.code_utils import custom_print


def get_feature_matrix_with_custom_features(dataset, feature_matrix_X):
    custom_print('Adding custom features to feature matrix...')
    # Adding custom features to turn the local descriptors into pseudo-global ones
    # Adding number of atoms as a feature
    db_sizes = np.array([len(struct) for struct in dataset])
    feature_matrix_X = np.concatenate(
        (feature_matrix_X, db_sizes.reshape(-1, 1)), axis=1
    )

    # Adding species fractions as features
    species = get_species_from_database(dataset)
    species_frac = np.zeros((len(dataset), len(species)))
    species_numbers = []
    for i, struct in enumerate(dataset):
        species_frac[i] = [struct.symbols.count(s) / len(struct) for s in species]
        species_numbers.append(
            [atomic_numbers[s] if struct.symbols.count(s) > 0 else 0 for s in species]
        )
    feature_matrix_X = np.concatenate((feature_matrix_X, species_frac), axis=1)

    # Adding species number as a feature
    species_numbers = np.array(species_numbers)
    feature_matrix_X = np.concatenate((feature_matrix_X, species_numbers), axis=1)

    # Adding cell volume as a feature
    db_cell_data = np.array(
        [
            [struct.get_volume()]
            + list(struct.cell.angles())
            + list(struct.cell.lengths())
            for struct in dataset
        ]
    )

    # Adding cell data to the feature matrix
    feature_matrix_X = np.concatenate((feature_matrix_X, db_cell_data), axis=1)

    # Adding atomic density as a feature
    density = np.array([len(struct) / struct.get_volume() for struct in dataset])
    feature_matrix_X = np.concatenate(
        (feature_matrix_X, density.reshape(-1, 1)), axis=1
    )

    # # Get rdf of each structure and add as features
    # from ase.geometry.analysis import get_rdf
    # new_feature_matrix = []
    # for i, struct in enumerate(dataset):
    #     # Compute RDF using ASE's get_rdf function
    #     # Parameters can be adjusted as needed
    #     try:
    #         rdf = get_rdf(struct, rmax=10.0, nbins=50, no_dists=True)
    #     except Exception:
    #         rdf = get_rdf(struct, rmax=0.5, nbins=50, no_dists=True)

    #     # Append RDF to feature matrix
    #     new_feature_matrix.append(np.concatenate((feature_matrix_X[i], rdf)))
    # new_feature_matrix = np.array(new_feature_matrix)

    # Standardizing features
    # This scales each feature (column) to have a mean of 0 and variance of 1
    # Important after adding the new features above, since they are in a completely
    # different range than the SOAP/MACE descriptors.
    scaler = StandardScaler()
    scaled_feature_matrix = scaler.fit_transform(feature_matrix_X)

    return scaled_feature_matrix


def get_vendi_score_db_rbf(
    dataset: list[Atoms],
    descriptor_type: str = 'soap',
    descriptor_settings: dict = None,
    mace_model_path: str = None,
    save_descriptors: bool = False,
    load_descriptors: bool = False,
    descriptors_path: str = 'descriptors.npy',
    sigma=None,
    k=1,
):
    """Computes the Vendi Score for a dataset of n structures using RBF kernel."""
    custom_print(
        f'Getting feature matrix for {len(dataset)} '
        f'structures using {descriptor_type.upper()}...'
    )

    _, feature_matrix_X = load_and_save_descriptors(
        dataset,
        descriptor_type,
        descriptor_settings,
        mace_model_path,
        save_descriptors,
        load_descriptors,
        descriptors_path,
    )

    custom_print(f'Feature matrix shape: {feature_matrix_X.shape}')

    return get_vendi_score(feature_matrix_X=feature_matrix_X, sigma=sigma, k=k)


def get_vendi_score(
    feature_matrix_X: np.ndarray,
    sigma: float | None = None,
    k: int = 1,
) -> float:
    """
    Lower time complexity than the alternative
    O(n^3) implementation above, but requires that the embeddings
    are available, and that the embedding dimension d is
    significantly smaller than n (d << n). This is usually the case
    when using MLIPs, and for large datasets.
    """
    n = feature_matrix_X.shape[0]

    print(f'{n=}')

    # Compute Pairwise Euclidean Distances
    # pdist returns a condensed distance matrix (efficient)
    dists = pdist(feature_matrix_X, metric='euclidean')

    # Heuristic for sigma
    # A common default is setting sigma to the median distance in the data.
    # However, it was found that using the mean k-NN distance works better
    # in practice.
    if sigma is None:
        # sigma = np.median(dists)

        # The Micro View (Average distance to k-th neighbor)
        # We need the full matrix to sort rows
        dist_matrix = squareform(dists)

        # Sort each row to find neighbors.
        # Column 0 is the point itself (dist=0), Column 1 is 1st NN, etc.
        # We take the column at index k.
        knn_distances = np.sort(dist_matrix, axis=1)[:, k]

        sigma = np.mean(knn_distances)
        custom_print(f'Heuristic (Mean {k}-NN Distance): sigma = {sigma:.4f}')
    else:
        custom_print(f'Using provided sigma: {sigma:.4f}')

    # 3. Apply RBF Kernel
    # Convert distances to similarity: K = exp(-d^2 / (2*sigma^2))
    K_condensed = np.exp(-(dists**2) / (2 * sigma**2))

    # Convert back to full n x n matrix for eigenvalue comp
    K = squareform(K_condensed)
    np.fill_diagonal(K, 1.0)  # Ensure diagonal is 1

    # 4. Compute Vendi Score (Standard method)
    K_norm = K / n
    eigenvalues = np.linalg.eigvalsh(K_norm)
    eigenvalues = np.clip(eigenvalues, 0, None)

    # 0 log 0 convention
    eps = 1e-20
    entropy = -np.sum(eigenvalues * np.log(eigenvalues + eps))

    return np.exp(entropy)


def _old_get_vendi_score(
    feature_matrix_X: np.ndarray,
) -> float:
    """
    Implementatio with a lower time complexity than the
    O(n^3) implementation above, but requires that the embeddings
    are available, and that the embedding dimension d is
    significantly smaller than n (d << n). This is usually the case
    when using MLIPs, and for large datasets.
    """
    # feature_matrix_X = np.vstack([desc.create(s) for s in dataset])

    # if use_custom_features:
    #     feature_matrix_X = get_feature_matrix_with_custom_features(
    #         dataset, feature_matrix_X
    #     )

    # Applying normalization.
    # Normalizing feature matrix before computing the covariance matrix.
    # This will make the inner product behave like a cosine similarity.
    feature_matrix_X = normalize(feature_matrix_X, norm='l2', axis=1)

    # Stack the list of feature vectors into a single (n, d) numpy array
    # n = number of samples
    # d = dimension of each feature vector
    n, d = feature_matrix_X.shape

    custom_print('Getting covariance matrix and computing eigenvalues...')

    # This replaces the creation of the n x n similarity matrix
    # The complexity is then O(d^2 * n), instead of O(n^3), which comes
    # from computing the eigenvalues of the n x n similarity matrix.
    if n < d:
        # Compute K = X @ X.T instead to save time
        covariance_matrix_C = feature_matrix_X @ feature_matrix_X.T
    else:
        # Compute the (d, d) covariance-like matrix C = X^T * X
        covariance_matrix_C = feature_matrix_X.T @ feature_matrix_X

    # Normalize similarity matrix by dividing by n
    normalized_matrix = covariance_matrix_C / n

    # Compute eigenvalues from the smaller matrix
    # The eigenvalues of K/n are the same as the eigenvalues of (X^T * X) / n
    # where K = X * X^T is the linear kernel matrix.
    eigenvalues = np.linalg.eigvalsh(normalized_matrix)

    # Ensure eigenvalues are non-negative (numerical stability)
    eigenvalues = np.clip(eigenvalues, a_min=0, a_max=None)

    # Compute Shannon entropy of eigenvalues
    # S_shannon = -\sum_{i=1}^n​ \lambda_i​ log(\lambda_i​)
    # Here, we force the convention 0 log(0) = 0 by adding a small epsilon.
    eps = 1e-20
    shannon_entropy = -np.sum(eigenvalues * np.log(eigenvalues + eps))

    # Compute Vendi Score
    # VS=\exp(S_shannon)
    vendi_score = np.exp(shannon_entropy)

    return vendi_score


def get_vendi_score_db_simple(
    dataset: list[Atoms],
    descriptor_type: str = 'soap',
    descriptor_settings: dict = None,
    mace_model_path: str = None,
    use_custom_features: bool = True,
    save_descriptors: bool = False,
    load_descriptors: bool = False,
    descriptors_path: str = 'descriptors.npy',
):
    """Computes the Vendi Score for a dataset of n structures."""
    custom_print(
        f'Getting feature matrix for {len(dataset)} '
        f'structures using {descriptor_type.upper()}...'
    )

    _, feature_matrix_X = load_and_save_descriptors(
        dataset,
        descriptor_type,
        descriptor_settings,
        mace_model_path,
        save_descriptors,
        load_descriptors,
        descriptors_path,
    )

    vendi_score = _old_get_vendi_score(feature_matrix_X)

    return vendi_score


def load_and_save_descriptors(
    dataset,
    descriptor_type,
    descriptor_settings,
    mace_model_path,
    save_descriptors,
    load_descriptors,
    descriptors_path,
    overwrite=False,
):
    if load_descriptors and pl.Path(descriptors_path).exists():
        feature_matrix_X = np.load(descriptors_path)
        custom_print(f'Loaded descriptors from {descriptors_path}...')
        descriptor_dict = {}

    else:
        outer_average = False
        if descriptor_settings.get('outer_average') is True:
            outer_average = descriptor_settings.pop('outer_average')
        descriptor_dict, feature_matrix_X, uuid_list = generate_descriptors(
            database=dataset,
            descriptor_type=descriptor_type,
            descriptor_settings=descriptor_settings,
            model_path=mace_model_path,
            outer_average_mace=outer_average,
        )
    if save_descriptors:
        custom_print(f'Saving descriptors to {descriptors_path}...')
        if pl.Path(descriptors_path).exists() and not overwrite:
            custom_print(
                f'File {descriptors_path} already exists. '
                'Use overwrite=True to overwrite.'
            )
        else:
            np.save(descriptors_path, feature_matrix_X)

    return descriptor_dict, feature_matrix_X


def tanimoto_distance(struct_1_descriptors, struct_2_descriptors):
    """
    Computes the Tanimoto distance.

    Also known as Jaccard distance.
    """
    tanimoto = 1 - np.sum(struct_1_descriptors * struct_2_descriptors) / (
        np.sum(np.maximum(struct_1_descriptors, struct_2_descriptors))
    )
    return tanimoto


def cosine_distance(v1, v2):
    """
    Calculates the cosine distance between two real-valued vectors.
    Returns a value between 0 (identical) and 1 (orthogonal for non-negative vectors).
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0

    similarity = dot_product / (norm_v1 * norm_v2)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def get_circles_metric_db(
    dataset: list[Atoms],
    distance_threshold: float,
    descriptor_settings: dict,
    distance_function=cosine_distance,
    use_custom_features: bool = True,
    save_descriptors: bool = False,
    load_descriptors: bool = False,
    descriptors_path: str = 'descriptors.npy',
    mace_model_path: str = None,
    descriptor_type: str = 'soap',
):
    """Computes the Circles Metric for the database.

    #Circles(S; d, t) := max |C| s.t. d(x, y) > t, ∀x ≠ y ∈ C
    """
    # max |C|
    # What is the maximum number of molecules I can pick from my database
    # such that every molecule I pick is far away form every other molecule
    # I've picked?

    # d(x, y) > t, ∀x ≠ y ∈ C
    # For every possible pair (x,y) of distinct molecules within the set C, the distance
    # between them must be greater than the threshold t.

    # Empty set
    n_structs = len(dataset)

    # Generate descriptors for the entire dataset
    if 'species' not in descriptor_settings:
        species = get_species_from_database(dataset)
        descriptor_settings['species'] = species

    _, feature_matrix_X = load_and_save_descriptors(
        dataset,
        descriptor_type,
        descriptor_settings,
        mace_model_path,
        save_descriptors,
        load_descriptors,
        descriptors_path,
    )

    feature_matrix_X = normalize(feature_matrix_X, norm='l2', axis=1)

    if use_custom_features:
        feature_matrix_X = get_feature_matrix_with_custom_features(
            dataset, feature_matrix_X
        )

    # Pre-compute the full n x n distance matrix for efficiency
    distance_matrix = np.zeros((n_structs, n_structs))
    for i in range(n_structs):
        for j in range(i + 1, n_structs):
            dist = distance_function(feature_matrix_X[i], feature_matrix_X[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Implement the greedy selection algorithm
    # Randomly reorder the indices of the dataset
    indices = np.arange(n_structs)
    np.random.shuffle(indices)

    # This set C will store the indices of the selected "circle centers"
    selected_indices = []

    for i in indices:
        is_far_enough = True
        # Check the distance to all previously selected molecules
        if selected_indices:
            distances_to_selected = distance_matrix[i, selected_indices]
            # If the minimum distance to an already selected point is not > t, reject it
            if np.min(distances_to_selected) <= distance_threshold:
                is_far_enough = False

        if is_far_enough:
            selected_indices.append(i)

    # The #Circles score is the number of molecules in our final set C
    return len(selected_indices)


class TestVendiScore(unittest.TestCase):
    """Unit tests for the Vendi Score computation."""

    def setUp(self):
        self.mock_loader_patcher = patch('__main__.load_and_save_descriptors')
        self.mock_loader = self.mock_loader_patcher.start()
        print('-' * 60)  # Separator for readability

    def tearDown(self):
        self.mock_loader_patcher.stop()

    def test_orthogonal_modes(self):
        """Test Scenario 1: Completely distinct items (Identity Matrix)."""
        X = np.eye(4)

        self.mock_loader.return_value = (None, X)
        dataset = [MagicMock()] * 4

        score = get_vendi_score_db_simple(dataset)

        print('Test: Orthogonal Modes (4 distinct items)')
        print('Expected: 4.0000')
        print(f'Actual:   {score:.4f}')

        self.assertAlmostEqual(score, 4.0, places=4)

    def test_identical_modes(self):
        """Test Scenario 2: All items are identical."""
        X = np.ones((4, 1))

        self.mock_loader.return_value = (None, X)
        dataset = [MagicMock()] * 4

        score = get_vendi_score_db_simple(dataset)

        print('Test: Identical Modes (4 identical items)')
        print('Expected: 1.0000')
        print(f'Actual:   {score:.4f}')

        self.assertAlmostEqual(score, 1.0, places=4)

    def test_two_distinct_groups(self):
        """Test Scenario 3: 4 items, 2 identical groups of 2."""
        X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        self.mock_loader.return_value = (None, X)
        dataset = [MagicMock()] * 4

        score = get_vendi_score_db_simple(dataset)

        print('Test: Two Distinct Groups (2 groups of 2)')
        print('Expected: 2.0000')
        print(f'Actual:   {score:.4f}')

        self.assertAlmostEqual(score, 2.0, places=4)

    def test_shape_and_color_overlap(self):
        """Test Scenario 4: Partial overlap (Shape & Color logic from paper)."""
        X = np.array(
            [[1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1]], dtype=float
        )

        self.mock_loader.return_value = (None, X)
        dataset = [MagicMock()] * 4

        score = get_vendi_score_db_simple(dataset)

        print('Test: Shape & Color Overlap (Sim=0.5)')
        print('Expected: 2.8284')
        print(f'Actual:   {score:.4f}')

        self.assertAlmostEqual(score, 2.8284, places=3)


def _to_dense(X):
    """Convert X to a dense numpy array if it is a sparse matrix, preserving dtype."""
    import scipy.sparse as sp

    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=X.dtype)
    return np.asarray(X)


def compute_heuristic_sigma(feature_matrix, sample_size=10000, seed=420):
    """
    Computes the median pairwise Euclidean distance of a subsample of the dataset.
    This is the standard heuristic for choosing the RBF kernel bandwidth (sigma).
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial.distance import pdist

    # Use the existing custom_print from your script
    from atlas.core.code_utils import custom_print

    n_samples = feature_matrix.shape[0]
    effective_size = min(sample_size, n_samples)

    custom_print(f'Tuning Sigma: Subsampling {effective_size} environments...')

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_samples, size=effective_size, replace=False)

    # Extract subset
    subset = feature_matrix[indices]

    # Convert to dense if it's a sparse matrix
    if sp.issparse(subset):
        subset = np.asarray(subset.todense(), dtype=np.float32)
    else:
        subset = np.asarray(subset, dtype=np.float32)

    # Compute condensed distance matrix
    dists = pdist(subset, metric='euclidean')

    # Calculate median and percentiles
    median_dist = float(np.median(dists))
    fine_grained = float(np.percentile(dists, 1))
    coarse = float(np.percentile(dists, 99))

    custom_print('\n' + '=' * 40)
    custom_print('      SIGMA (RBF BANDWIDTH) HEURISTICS      ')
    custom_print('=' * 40)
    custom_print(f' Fine-grained                   : {fine_grained:.4f}')
    custom_print(f' Median                         : {median_dist:.4f}')
    custom_print(f' Coarse                         : {coarse:.4f}')
    custom_print('=' * 40 + '\n')

    return (fine_grained, median_dist, coarse)


def get_vendi_score_subsampling(
    feature_matrix, subset_size=10000, n_iterations=4, sigma=0.1624, k=1
):
    scores = []

    custom_print('Computing Vendi Score via Subsampling...')

    rng = np.random.default_rng()

    if sigma is None:
        custom_print(f'Getting {k}-NN distances to compute heuristic sigma...')

        # Randomly subsampling for sigma

        # Picking a subset of the data to compute the heuristic sigma
        # without repeating
        indices = rng.choice(
            feature_matrix.shape[0],
            size=min(subset_size, feature_matrix.shape[0]),
            replace=False,
        )
        X_sub = _to_dense(feature_matrix[indices])

        # Getting distances
        sub_dists = pdist(X_sub, metric='euclidean')

        # Average distance to k-th neighbor
        # We need the full matrix to sort rows
        dist_matrix = squareform(sub_dists)

        # Sort each row to find neighbors.
        # Column 0 is the point itself (dist=0), Column 1 is 1st NN, etc.
        # We take the column at index k.
        knn_distances = np.sort(dist_matrix, axis=1)[:, k]

        sigma = np.mean(knn_distances)
        custom_print(f'Heuristic (Mean {k}-NN Distance): sigma = {sigma:.4f}')
    else:
        custom_print(f'Using provided sigma: {sigma:.4f}')

    # feature_matrix may be sparse; we densify only the subsets below
    if subset_size > feature_matrix.shape[0]:
        custom_print(
            'Subset size larger than dataset size. Computing on full dataset...'
        )

        # Compute RBF Kernel (Exact)
        dists = pdist(_to_dense(feature_matrix), metric='euclidean')
        K_condensed = np.exp(-(dists**2) / (2 * sigma**2))
        K = squareform(K_condensed)
        np.fill_diagonal(K, 1.0)

        # Eigenvalues
        K_norm = K / feature_matrix.shape[0]
        eigvals = np.linalg.eigvalsh(K_norm)
        eigvals = np.clip(eigvals, 1e-20, None)

        # Entropy
        entropy = -np.sum(eigvals * np.log(eigvals))
        scores = [np.exp(entropy)]

    else:
        custom_print(
            f'Computing Vendi Score on {n_iterations} subsets of size {subset_size}...'
        )

        for i in range(n_iterations):
            # Subsample
            rng = np.random.default_rng(i)  # Different seed per iteration
            indices = rng.choice(
                feature_matrix.shape[0], size=subset_size, replace=False
            )
            X_sub = _to_dense(feature_matrix[indices])

            # Compute RBF Kernel (Exact)
            dists = pdist(X_sub, metric='euclidean')
            K_condensed = np.exp(-(dists**2) / (2 * sigma**2))
            K = squareform(K_condensed)
            np.fill_diagonal(K, 1.0)

            # Eigenvalues
            K_norm = K / subset_size
            eigvals = np.linalg.eigvalsh(K_norm)
            eigvals = np.clip(eigvals, 1e-20, None)

            # Entropy
            entropy = -np.sum(eigvals * np.log(eigvals))
            score = np.exp(entropy)

            scores.append(score)
            custom_print(f'  Iteration {i + 1}: {score:.2f}')

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    custom_print(f'Mean Vendi Score: {mean_score:.2f} ± {std_score:.2f}', 'done')

    return scores, mean_score, std_score


if __name__ == '__main__':
    # We define a dummy load_and_save_descriptors for the mock to patch
    def load_and_save_descriptors(*args):
        pass

    unittest.main()
