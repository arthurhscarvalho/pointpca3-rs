use kd_tree;
use pointpca2_rs::knn_search;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[inline(always)]
fn compute_bounding_box(points_a: &Vec<[f64; 3]>) -> (f64, f64, f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for p in points_a.iter() {
        if p[0] < min_x {
            min_x = p[0];
        }
        if p[1] < min_y {
            min_y = p[1];
        }
        if p[2] < min_z {
            min_z = p[2];
        }
        if p[0] > max_x {
            max_x = p[0];
        }
        if p[1] > max_y {
            max_y = p[1];
        }
        if p[2] > max_z {
            max_z = p[2];
        }
    }
    (min_x, min_y, min_z, max_x, max_y, max_z)
}

pub fn voxel_based_sampling(
    kd_tree_a: &kd_tree::KdIndexTree3<[f64; 3]>,
    points_a: &Vec<[f64; 3]>,
    search_size: usize,
) -> Vec<usize> {
    let nrows = points_a.len();
    if nrows == 0 {
        return Vec::new();
    }
    let (min_x, min_y, min_z, max_x, max_y, max_z) = compute_bounding_box(points_a);
    let dx = (max_x - min_x).max(1e-9);
    let dy = (max_y - min_y).max(1e-9);
    let dz = (max_z - min_z).max(1e-9);
    let volume = dx * dy * dz;
    // 2) Define a voxelization distance (alpha)
    // Use average inter-point spacing approximation: alpha ≈ cbrt(volume / n)
    let alpha = (volume / nrows as f64).cbrt().max(1.0);
    // Shared visited state across threads
    let visited: Arc<Vec<AtomicBool>> =
        Arc::new((0..nrows).map(|_| AtomicBool::new(false)).collect());
    let seeds_acc: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    // Helper to compute voxel key for a point
    let voxel_key = |p: &[f64; 3]| -> [i32; 3] {
        let ix = ((p[0] - min_x) / alpha).floor() as i32;
        let iy = ((p[1] - min_y) / alpha).floor() as i32;
        let iz = ((p[2] - min_z) / alpha).floor() as i32;
        [ix, iy, iz]
    };
    // Iterate passes until all points are visited
    loop {
        // Gather current unvisited points
        let mut unvisited_indices: Vec<usize> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            if !visited[i].load(Ordering::Relaxed) {
                unvisited_indices.push(i);
            }
        }
        if unvisited_indices.is_empty() {
            break;
        }
        // 3) Build voxel map for the unvisited set, selecting one representative per voxel
        // Criterion: closest to the voxel center
        let mut voxel_to_candidate: HashMap<[i32; 3], (usize, f64)> = HashMap::new();
        for &idx in unvisited_indices.iter() {
            let p = &points_a[idx];
            let key = voxel_key(p);
            let center_x = min_x + (key[0] as f64 + 0.5) * alpha;
            let center_y = min_y + (key[1] as f64 + 0.5) * alpha;
            let center_z = min_z + (key[2] as f64 + 0.5) * alpha;
            let dx = p[0] - center_x;
            let dy = p[1] - center_y;
            let dz = p[2] - center_z;
            let dist2 = dx * dx + dy * dy + dz * dz;
            match voxel_to_candidate.get_mut(&key) {
                Some((best_idx, best_dist2)) => {
                    if dist2 < *best_dist2 || (dist2 == *best_dist2 && idx < *best_idx) {
                        *best_idx = idx;
                        *best_dist2 = dist2;
                    }
                }
                None => {
                    voxel_to_candidate.insert(key, (idx, dist2));
                }
            }
        }
        // Collect unique candidate indices for this pass
        let candidates: Vec<usize> = voxel_to_candidate.values().map(|(idx, _)| *idx).collect();
        // 4) For each voxel (candidate), compute KNN and mark visited, in parallel
        candidates.par_iter().for_each(|&i| {
            // Attempt to claim this seed (avoid duplicates across overlapping neighborhoods)
            if visited[i]
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                // Record seed
                {
                    let mut guard = seeds_acc.lock().unwrap();
                    guard.push(i);
                }
                // Mark its neighborhood
                let knn_indices_a = knn_search::nearest_n(&kd_tree_a, &points_a[i], search_size);
                for &idx in knn_indices_a.iter() {
                    if idx < nrows {
                        visited[idx].store(true, Ordering::Relaxed);
                    }
                }
            }
        });
        // 5) If after the pass there are still unvisited points, the loop continues
        // 6) Loop repeats until all points are visited
    }
    let seeds = Arc::try_unwrap(seeds_acc).unwrap().into_inner().unwrap();
    seeds
}
