extern crate kd_tree;
extern crate nalgebra as na;
extern crate numpy;
extern crate pointpca2_rs;
extern crate rayon;

mod pointpca3;
mod utils;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use utils::{build_thread_pool, to_vec};

#[pyfunction]
fn compute_pointpca3<'py>(
    py: Python<'py>,
    points_a: PyReadonlyArray2<'py, f64>,
    colors_a: PyReadonlyArray2<'py, u8>,
    points_b: PyReadonlyArray2<'py, f64>,
    colors_b: PyReadonlyArray2<'py, u8>,
    search_size: usize,
    max_workers: usize,
    verbose: bool,
) -> Bound<'py, PyArray1<f64>> {
    let points_a = to_vec(points_a);
    let colors_a = to_vec(colors_a);
    let points_b = to_vec(points_b);
    let colors_b = to_vec(colors_b);
    let pool = build_thread_pool(max_workers);
    let pooled_predictors = py.allow_threads(|| {
        pool.install(|| {
            pointpca3::compute_pointpca3(
                points_a,
                colors_a,
                points_b,
                colors_b,
                search_size,
                verbose,
            )
        })
    });
    let py_array = PyArray1::from_iter(py, pooled_predictors.iter().cloned());
    py_array
}

#[pymodule(name = "pointpca3")]
fn init_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_pointpca3, m)?)?;
    Ok(())
}
