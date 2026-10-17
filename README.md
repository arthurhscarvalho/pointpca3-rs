# pointpca3-rs

## Purpose

Rust implementation of PointPCA3 for full-reference colored point-cloud quality assessment. This implementation uses Voxelization-based Spatial Sampling (VoSS) to select reference seeds, then pools local geometry and color comparisons into 40 predictors. The output is a feature vector for subsequent regression, rather than a predicted perceptual quality score.

## Related projects

The [pointpca3-pylib](https://github.com/arthurhscarvalho/pointpca3-pylib) project exposes this backend to Python. This crate depends on [pointpca2-rs](https://github.com/arthurhscarvalho/pointpca2-rs), fetched by Cargo from Git tag `v1.5.1`, for predictor computation, utilities, and the example PLY reader. [MAPPE](https://github.com/arthurhscarvalho/mappe) uses these point-cloud features in its mesh quality assessment pipeline.

## Requirements

- A current stable Rust toolchain with Cargo and a native linker/C compiler. The crate uses Rust edition 2021.
- Git and network access to fetch dependencies.
- Reference and distorted point clouds with XYZ coordinates and RGB colors.

## Build

```bash
git clone https://github.com/arthurhscarvalho/pointpca3-rs.git
cd pointpca3-rs
cargo build --release
```

Dependencies are declared in [Cargo.toml](Cargo.toml). No sibling repository checkout is required for this build.

## Usage

### Rust API

The public function in [src/lib.rs](src/lib.rs) is:

```rust
pub fn compute_pointpca3(
    points_a: Vec<[f64; 3]>,
    colors_a: Vec<[u8; 3]>,
    points_b: Vec<[f64; 3]>,
    colors_b: Vec<[u8; 3]>,
    search_size: usize,
    verbose: bool,
) -> nalgebra::Matrix1xX<f64>
```

`a` is the reference cloud and `b` is the distorted cloud. Coordinates and colors must have matching lengths within each cloud; the two clouds may contain different numbers of points. Supply RGB colors in `[0, 255]`. Choose a positive neighborhood size no larger than the number of distinct points in either cloud after duplicate merging. The example uses `search_size = 81`.

The function consumes the input vectors and returns a row matrix of shape `1 × 40`. It uses Rayon for parallel processing. Set `RAYON_NUM_THREADS` before execution to control the global Rayon thread pool if needed.

### Bundled executable

[src/main.rs](src/main.rs) loads two PLY files, computes the predictors, and prints them. Replace `<path-to-reference>` and `<path-to-distorted>` in that file with actual file paths, then run from the repository root:

```bash
cargo run --release
```

The executable does not accept point-cloud paths as command-line arguments. Its PLY loader expects vertex properties `x`, `y`, `z`, `red`, `green`, and `blue`; it is an experimental file-reading utility. For other input formats, load the arrays separately and call the Rust API.

## Publications

The first publication describes PointPCA3. The second describes the PointPCA2-RS implementation on which it builds.

1. Arthur Henrique Silva Carvalho and Pedro Garcia Freitas. “PointPCA3: Scalable Point Cloud Quality Assessment via Voxelization-based Spatial Sampling.” *2026 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB)*, 2026. Accepted and presented; IEEE Xplore publication pending. [Conference proceedings series](https://ieeexplore.ieee.org/xpl/conhome/1001845/all-proceedings).
2. Arthur Henrique Silva Carvalho and Pedro Garcia Freitas. “PointPCA2-RS: PointPCA2 with Resource Saving.” *XLIII Brazilian Symposium on Telecommunications and Signal Processing (SBrT)*, Natal, Brazil, 2025. [Paper](https://biblioteca.sbrt.org.br/articlefile/4834.pdf).

```bibtex
@inproceedings{carvalho2026pointpca3,
  author    = {Carvalho, Arthur Henrique Silva and Freitas, Pedro Garcia},
  title     = {{PointPCA3}: Scalable Point Cloud Quality Assessment via Voxelization-based Spatial Sampling},
  booktitle = {2026 IEEE International Symposium on Broadband Multimedia Systems and Broadcasting (BMSB)},
  year      = {2026},
  note      = {Accepted and presented; IEEE Xplore publication pending}
}

@inproceedings{carvalho2025pointpca2rs,
  author    = {Carvalho, Arthur Henrique Silva and Freitas, Pedro Garcia},
  title     = {{PointPCA2-RS}: {PointPCA2} with Resource Saving},
  booktitle = {XLIII Brazilian Symposium on Telecommunications and Signal Processing (SBrT)},
  address   = {Natal, Brazil},
  year      = {2025},
  url       = {https://biblioteca.sbrt.org.br/articlefile/4834.pdf}
}
```

## License

[MIT License](LICENSE).
