use na::DMatrix;
use pointpca2_rs::{knn_search, pca, utils};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

const FEATURES_DIMENSION: usize = 42;

fn compute_seeds(
    kd_tree_a: &kd_tree::KdIndexTree3<[f64; 3]>,
    points_a: &Vec<[f64; 3]>,
    search_size: usize,
) -> Vec<usize> {
    let nrows = points_a.len();
    let mut visited = vec![false; nrows];
    let mut seeds: Vec<usize> = Vec::new();
    for i in 0..nrows {
        if visited[i] {
            continue;
        }
        seeds.push(i);
        visited[i] = true;
        let knn_indices_a = knn_search::nearest_n(&kd_tree_a, &points_a[i], search_size);
        for &idx in knn_indices_a.iter() {
            if idx < nrows {
                visited[idx] = true;
            }
        }
    }
    seeds
}

pub fn compute_features(
    points_a: Vec<[f64; 3]>,
    colors_a: Vec<[u8; 3]>,
    points_b: Vec<[f64; 3]>,
    colors_b: Vec<[u8; 3]>,
    search_size: usize,
) -> DMatrix<f64> {
    let kd_tree_a = knn_search::build_tree(&points_a);
    let kd_tree_b = knn_search::build_tree(&points_b);
    // Phase 1: Greedy-by-fixed-order deterministic selection (sequential)
    let seeds = compute_seeds(&kd_tree_a, &points_a, search_size);
    // Phase 2: Parallel feature computation for selected seeds; fixed-order output
    let mut rows = vec![[0.0f64; FEATURES_DIMENSION]; seeds.len()];
    rows.par_iter_mut().enumerate().for_each(|(row_idx, row)| {
        let i = seeds[row_idx];
        let knn_indices_a = knn_search::nearest_n(&kd_tree_a, &points_a[i], search_size);
        let knn_indices_b = knn_search::nearest_n(&kd_tree_b, &points_a[i], search_size);
        let (sl_points_a, sl_colors_a) =
            utils::slice_from_knn_indices(&points_a, &colors_a, &knn_indices_a, search_size);
        let (sl_points_b, sl_colors_b) =
            utils::slice_from_knn_indices(&points_b, &colors_b, &knn_indices_b, search_size);
        let eigenvectors_a = pca::compute_pca(&sl_points_a);
        let sl_points_a_mean = sl_points_a.row_mean();
        let projection_a_to_a =
            utils::subtract_row_from_matrix(&sl_points_a, &sl_points_a_mean) * &eigenvectors_a;
        let projection_b_to_a =
            utils::subtract_row_from_matrix(&sl_points_b, &sl_points_a_mean) * &eigenvectors_a;
        let mean_a = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a).row_mean();
        let mean_b = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b).row_mean();
        let proj_colors_a_concat = utils::concatenate_columns(&projection_a_to_a, &sl_colors_a);
        let proj_colors_b_concat = utils::concatenate_columns(&projection_b_to_a, &sl_colors_b);
        let mean_deviation_a = utils::subtract_row_from_matrix(&proj_colors_a_concat, &mean_a);
        let mean_deviation_b = utils::subtract_row_from_matrix(&proj_colors_b_concat, &mean_b);
        let variance_a = mean_deviation_a.map(|x| x.powi(2)).row_mean();
        let variance_b = mean_deviation_b.map(|x| x.powi(2)).row_mean();
        let covariance_ab = mean_deviation_a.component_mul(&mean_deviation_b).row_mean();
        let eigenvectors_b = pca::compute_pca(&projection_b_to_a).transpose();
        // Fill row in fixed order
        let mut c = 0;
        for v in projection_a_to_a.row(0).iter() {
            row[c] = *v;
            c += 1;
        }
        for v in projection_b_to_a.row(0).iter() {
            row[c] = *v;
            c += 1;
        }
        for v in mean_a.columns(3, 3).iter() {
            row[c] = *v;
            c += 1;
        }
        for v in mean_b.iter() {
            row[c] = *v;
            c += 1;
        }
        for v in variance_a.iter() {
            row[c] = *v;
            c += 1;
        }
        for v in variance_b.iter() {
            row[c] = *v;
            c += 1;
        }
        for v in covariance_ab.iter() {
            row[c] = *v;
            c += 1;
        }
        for v in eigenvectors_b.row(0).iter() {
            row[c] = *v;
            c += 1;
        }
        for v in eigenvectors_b.row(1).iter() {
            row[c] = *v;
            c += 1;
        }
        for v in eigenvectors_b.row(2).iter() {
            row[c] = *v;
            c += 1;
        }
    });
    // Flatten rows into a single buffer
    let mut features = Vec::with_capacity(rows.len() * FEATURES_DIMENSION);
    for row in rows.into_iter() {
        features.extend_from_slice(&row);
    }
    DMatrix::from_row_slice(
        features.len() / FEATURES_DIMENSION,
        FEATURES_DIMENSION,
        &features,
    )
}
