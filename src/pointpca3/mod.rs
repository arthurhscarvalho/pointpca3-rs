use pointpca2_rs::{predictors, utils};

mod features;
mod preprocessing;
mod voxelization;

pub fn compute_pointpca3(
    points_a: Vec<[f64; 3]>,
    colors_a: Vec<[u8; 3]>,
    points_b: Vec<[f64; 3]>,
    colors_b: Vec<[u8; 3]>,
    search_size: usize,
    verbose: bool,
) -> na::Matrix1xX<f64> {
    utils::print_if_verbose("Preprocessing", &verbose);
    let t0 = std::time::Instant::now();
    let (points_a, colors_a) = preprocessing::preprocess_point_cloud(points_a, colors_a);
    let (points_b, colors_b) = preprocessing::preprocess_point_cloud(points_b, colors_b);
    let t1 = std::time::Instant::now();
    utils::print_if_verbose("Computing local features", &verbose);
    let local_features =
        features::compute_features(points_a, colors_a, points_b, colors_b, search_size);
    let t2 = std::time::Instant::now();
    utils::print_if_verbose("Computing predictors", &verbose);
    let predictors_result = predictors::compute_predictors(local_features);
    let t3 = std::time::Instant::now();
    utils::print_if_verbose("Total time", &verbose);
    utils::print_if_verbose(&format!("Preprocessing: {:?} seconds", t1 - t0), &verbose);
    utils::print_if_verbose(
        &format!("Computing local features: {:?} seconds", t2 - t1),
        &verbose,
    );
    utils::print_if_verbose(
        &format!("Computing predictors: {:?} seconds", t3 - t2),
        &verbose,
    );
    predictors_result
}
