use super::voxelization::voxel_based_sampling;
use na::DMatrix;
use pointpca2_rs::{knn_search, pca, utils};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

const FEATURES_DIMENSION: usize = 42;

#[inline(always)]
fn copy_iter<'a, I>(row: &mut [f64], idx: &mut usize, iter: I)
where
    I: IntoIterator<Item = &'a f64>,
{
    for &value in iter {
        row[*idx] = value;
        *idx += 1;
    }
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
    let t0 = std::time::Instant::now();
    let seeds = voxel_based_sampling(&kd_tree_a, &points_a, search_size);
    let t1 = std::time::Instant::now();
    println!("Voxel-based sampling time: {:?}", t1 - t0);
    println!("Number of seeds: {}", seeds.len());
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
        let mut idx = 0;
        copy_iter(row, &mut idx, projection_a_to_a.row(0).iter());
        copy_iter(row, &mut idx, projection_b_to_a.row(0).iter());
        copy_iter(row, &mut idx, mean_a.columns(3, 3).iter());
        copy_iter(row, &mut idx, mean_b.iter());
        copy_iter(row, &mut idx, variance_a.iter());
        copy_iter(row, &mut idx, variance_b.iter());
        copy_iter(row, &mut idx, covariance_ab.iter());
        copy_iter(row, &mut idx, eigenvectors_b.row(0).iter());
        copy_iter(row, &mut idx, eigenvectors_b.row(1).iter());
        copy_iter(row, &mut idx, eigenvectors_b.row(2).iter());
    });
    DMatrix::from_row_iterator(rows.len(), FEATURES_DIMENSION, rows.into_iter().flatten())
}
