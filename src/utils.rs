use numpy::PyReadonlyArray2;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;

pub fn to_vec<T>(x: PyReadonlyArray2<T>) -> Vec<[T; 3]>
where
    T: numpy::Element + Copy,
{
    let array = x.as_array();
    let n = array.shape()[0];
    let mut vec = Vec::with_capacity(n);
    for row in array.rows() {
        vec.push([row[0], row[1], row[2]]);
    }
    vec
}

pub fn build_thread_pool(max_workers: usize) -> ThreadPool {
    let pool = ThreadPoolBuilder::new()
        .num_threads(max_workers)
        .build()
        .expect("Failed to build thread pool");
    pool
}
