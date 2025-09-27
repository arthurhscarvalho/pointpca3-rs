use rayon::slice::ParallelSliceMut;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct BitPoint3(u64, u64, u64);

impl BitPoint3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self(x.to_bits(), y.to_bits(), z.to_bits())
    }

    fn to_point(&self) -> [f64; 3] {
        [
            f64::from_bits(self.0),
            f64::from_bits(self.1),
            f64::from_bits(self.2),
        ]
    }
}

#[inline(always)]
fn rgb_to_yuv(rgb: [u8; 3]) -> [u8; 3] {
    // BT.709 full-range, integer fixed-point (scale = 1<<16)
    // Y  = 0.2126 R + 0.7152 G + 0.0722 B
    // Cb = -0.1146 R - 0.3854 G + 0.5000 B + 128
    // Cr =  0.5000 R - 0.4542 G - 0.0468 B + 128
    const SCALE: i32 = 1 << 16;
    const HALF: i32 = 1 << 15;
    // Y coefficients
    const YR: i32 = 13933;
    const YG: i32 = 46871;
    const YB: i32 = 4732;
    // Cb coefficients
    const UO: i32 = 128 * SCALE;
    const UR: i32 = -7510;
    const UG: i32 = -25258;
    const UB: i32 = 32768;
    // Cr coefficients
    const VO: i32 = 128 * SCALE;
    const VR: i32 = 32768;
    const VG: i32 = -29766;
    const VB: i32 = -3067;
    // Convert to integer fixed-point
    let r = rgb[0] as i32;
    let g = rgb[1] as i32;
    let b = rgb[2] as i32;
    // Compute YUV
    let y = ((YR * r + YG * g + YB * b + HALF) >> 16).clamp(0, 255) as u8;
    let u = ((UR * r + UG * g + UB * b + UO + HALF) >> 16).clamp(0, 255) as u8;
    let v = ((VR * r + VG * g + VB * b + VO + HALF) >> 16).clamp(0, 255) as u8;
    [y, u, v]
}

pub fn preprocess_point_cloud(
    points: Vec<[f64; 3]>,
    colors: Vec<[u8; 3]>,
) -> (Vec<[f64; 3]>, Vec<[u8; 3]>) {
    let n = points.len().min(colors.len());
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // Zip into (BitPoint3, rgb) pairs
    let mut pairs: Vec<(BitPoint3, [u8; 3])> = Vec::with_capacity(n);
    for i in 0..n {
        pairs.push((
            BitPoint3::new(points[i][0], points[i][1], points[i][2]),
            colors[i],
        ));
    }
    // Parallel sort by BitPoint3 key (lex over u64 triples)
    pairs.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
    // One pass scan to aggregate sums and emit means
    let mut points_result: Vec<[f64; 3]> = Vec::new();
    let mut colors_result: Vec<[u8; 3]> = Vec::new();
    let mut i = 0usize;
    while i < pairs.len() {
        let key = &pairs[i].0;
        let mut sum_r: u64 = 0;
        let mut sum_g: u64 = 0;
        let mut sum_b: u64 = 0;
        let mut cnt: u64 = 0;
        let mut j = i;
        while j < pairs.len() && pairs[j].0 == *key {
            let c = pairs[j].1;
            sum_r += c[0] as u64;
            sum_g += c[1] as u64;
            sum_b += c[2] as u64;
            cnt += 1;
            j += 1;
        }
        let avg_r = ((sum_r + (cnt >> 1)) / cnt) as u8;
        let avg_g = ((sum_g + (cnt >> 1)) / cnt) as u8;
        let avg_b = ((sum_b + (cnt >> 1)) / cnt) as u8;
        let yuv = rgb_to_yuv([avg_r, avg_g, avg_b]);
        points_result.push(key.to_point());
        colors_result.push(yuv);
        i = j;
    }
    (points_result, colors_result)
}
