use na::DMatrix;
use pointpca2_rs::{knn_search, pca, utils};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

const FEATURES_DIMENSION: usize = 42;

pub fn compute_features(
    points_a: Vec<[f64; 3]>,
    colors_a: Vec<[u8; 3]>,
    points_b: Vec<[f64; 3]>,
    colors_b: Vec<[u8; 3]>,
    search_size: usize,
) -> DMatrix<f64> {
    let kd_tree_a = knn_search::build_tree(&points_a);
    let kd_tree_b = knn_search::build_tree(&points_b);
    let nrows = points_a.len();

    // Use atomic booleans for thread-safe, deterministic visited tracking
    // Memory usage: nrows * 1 byte (1MB for 1M elements)
    let visited = (0..nrows)
        .map(|_| AtomicBool::new(false))
        .collect::<Vec<_>>();

    let features: Mutex<Vec<f64>> = Mutex::new(Vec::new());

    (0..nrows).into_par_iter().for_each(|i| {
        // Atomically check and set visited status in a single operation
        if visited[i]
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            // Already visited, skip
            return;
        }
        // Get the nearest neighbors
        let knn_indices_a = knn_search::nearest_n(&kd_tree_a, &points_a[i], search_size);
        let knn_indices_b = knn_search::nearest_n(&kd_tree_b, &points_a[i], search_size);
        // Mark all neighbors as visited atomically
        for &idx in knn_indices_a.iter() {
            if idx < nrows {
                visited[idx].store(true, Ordering::SeqCst);
            }
        }
        // Slice points and colors from their respective knn indices
        let (sl_points_a, sl_colors_a) =
            utils::slice_from_knn_indices(&points_a, &colors_a, &knn_indices_a, search_size);
        let (sl_points_b, sl_colors_b) =
            utils::slice_from_knn_indices(&points_b, &colors_b, &knn_indices_b, search_size);
        // Principal components of reference data (new orthonormal basis)
        let eigenvectors_a = pca::compute_pca(&sl_points_a);
        // Project reference and distorted data onto the new orthonormal basis
        let sl_points_a_mean = sl_points_a.row_mean();
        let projection_a_to_a =
            utils::subtract_row_from_matrix(&sl_points_a, &sl_points_a_mean) * &eigenvectors_a;
        let projection_b_to_a =
            utils::subtract_row_from_matrix(&sl_points_b, &sl_points_a_mean) * &eigenvectors_a;
        // Mean values for projected geometric data and texture data
        let mean_a = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a).row_mean();
        let mean_b = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b).row_mean();
        let proj_colors_a_concat = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a);
        let proj_colors_b_concat = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b);
        // Deviation from mean
        let mean_deviation_a = utils::subtract_row_from_matrix(&proj_colors_a_concat, &mean_a);
        let mean_deviation_b = utils::subtract_row_from_matrix(&proj_colors_b_concat, &mean_b);
        // Variances and covariance
        let variance_a = mean_deviation_a.map(|x| x.powi(2)).row_mean();
        let variance_b = mean_deviation_b.map(|x| x.powi(2)).row_mean();
        let covariance_ab = mean_deviation_a.component_mul(&mean_deviation_b).row_mean();
        // Principal components of projected distorted data
        let eigenvectors_b = pca::compute_pca(&projection_b_to_a).transpose();
        // Build the row for this point
        let mut row_values: Vec<f64> = Vec::with_capacity(FEATURES_DIMENSION);
        row_values.extend(projection_a_to_a.row(0).iter()); // 0..3
        row_values.extend(projection_b_to_a.row(0).iter()); // 3..6
        row_values.extend(mean_a.columns(3, 3).iter()); // 6..9
        row_values.extend(mean_b.iter()); // 9..15
        row_values.extend(variance_a.iter()); // 15..21
        row_values.extend(variance_b.iter()); // 21..27
        row_values.extend(covariance_ab.iter()); // 27..33
        row_values.extend(eigenvectors_b.row(0).iter()); // 33..36
        row_values.extend(eigenvectors_b.row(1).iter()); // 36..39
        row_values.extend(eigenvectors_b.row(2).iter()); // 39..42
        let mut features = features.lock().unwrap();
        features.extend(row_values);
    });
    let features = features.into_inner().unwrap();
    let nrows = features.len() / FEATURES_DIMENSION;
    DMatrix::from_row_slice(nrows, FEATURES_DIMENSION, &features)
}
