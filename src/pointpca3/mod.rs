use pointpca2_rs::predictors;
use pointpca2_rs::preprocessing;
use pointpca2_rs::utils;

mod features;

pub fn compute_pointpca3<'a>(
    points_a: Vec<[f64; 3]>,
    colors_a: Vec<[u8; 3]>,
    points_b: Vec<[f64; 3]>,
    colors_b: Vec<[u8; 3]>,
    search_size: usize,
    verbose: bool,
) -> na::Matrix1xX<f64> {
    utils::print_if_verbose("Preprocessing", &verbose);
    let (points_a, colors_a) = preprocessing::preprocess_point_cloud(points_a, colors_a);
    let (points_b, colors_b) = preprocessing::preprocess_point_cloud(points_b, colors_b);
    utils::print_if_verbose("Computing local features", &verbose);
    let local_features =
        features::compute_features(points_a, colors_a, points_b, colors_b, search_size);
    utils::print_if_verbose("Computing predictors", &verbose);
    let predictors_result = predictors::compute_predictors(local_features);
    predictors_result
}
