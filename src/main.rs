use polars::prelude::*;
use std::error::Error;
use nalgebra::{DMatrix, DVector, SVector, SMatrix};
use ndarray_stats::CorrelationExt;
use std::sync::LazyLock;
use Rust_MCS::*;


const N_ASSETS: usize = 206;
const CSV_PATH: &str = "rets.csv";
const MIN_NON_ZERO_WEIGHT: f64 = 0.0; // 1% (weights can be 0 or [MIN_NON_ZERO_WEIGHT; MAX_WEIGHT]); if set to 0.0 -> no effect
const MAX_WEIGHT: f64 = 0.1; // 10%


fn load_mean_cov_matrices(csv_path: &str) ->
Result<(
    DVector<f64>,
    DMatrix<f64>
), Box<dyn Error>>
{
    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(csv_path.into()))?
        .finish()
        .unwrap();

    // Drop the "date" column
    let nd_matrix = df.drop("date")?.to_ndarray::<Float64Type>(IndexOrder::C)?;

    // Calculate mean for each column
    let hist_mean = nd_matrix.mean_axis(ndarray::Axis(0)).unwrap();

    // Calculate the covariance matrix
    let cov_matrix = nd_matrix.t().cov(1.).unwrap();

    let hist_mean = DVector::<f64>::from_iterator(N_ASSETS, hist_mean.into_iter());
    let cov_matrix = DMatrix::<f64>::from_iterator(N_ASSETS, N_ASSETS, cov_matrix.into_iter());

    println!("Historical Mean (first 5 values):");
    println!("{:.6}", hist_mean.rows(0, 5));
    println!("{:.6}", hist_mean.rows(N_ASSETS - 5, 5));

    println!("Covariance Matrix (first 5 rows and cols):");
    println!("{:.6}", cov_matrix.view((0, 0), (5, 5)));

    Ok((hist_mean, cov_matrix))
}


fn sharpe_ratio(weights: &SVector<f64, N_ASSETS>) -> f64 {
    let weights_scaled = modify_weights(weights);
    let port_ret = weights_scaled.dot(&(*HIST_MEAN_COV).0);
    let port_std = weights_scaled.dot(&(&(*HIST_MEAN_COV).1 * weights_scaled)).powf(0.5);
    let sharpe_ratio = -port_ret / port_std;

    sharpe_ratio
}


pub fn modify_weights(weights: &SVector<f64, N_ASSETS>) -> SVector<f64, N_ASSETS> {
    let mut infinite_loop_counter = 0_usize;

    // sets weights that are smaller than MIN_WEIGHT*100% to 0
    let mut scaled_weights = weights.map(|weight| if weight < MIN_NON_ZERO_WEIGHT { 0.0 } else { weight });

    if scaled_weights.max() == 0.0 ||
        scaled_weights.iter().filter(|&&weight| weight > f64::EPSILON).count() <= 10 ||
        scaled_weights.iter().find(|&&weight| weight.is_nan()).is_some() {
        return SVector::<f64, N_ASSETS>::repeat(1.0 / N_ASSETS as f64);
    }

    scaled_weights.scale_mut(1.0_f64 / scaled_weights.sum());
    // Now weights sum up to 1, but some weights can be > MAX_WEIGHT.

    while scaled_weights.max() > MAX_WEIGHT + f64::EPSILON {
        infinite_loop_counter += 1;
        let scaling_factor = MAX_WEIGHT / scaled_weights.max();
        scaled_weights.scale_mut(scaling_factor);
        // max is exactly MAX_WEIGHT now, but sum is not

        // need_to_allocate: positive as scaling_factor < 1.0
        let need_to_allocate = 1.0 - scaling_factor;
        let sum_of_weights_to_modif = scaled_weights.fold(0.0, |sum, weight| if (weight >= MAX_WEIGHT - f64::EPSILON) || (weight == 0.0) { sum } else { sum + weight });

        for weight in scaled_weights.iter_mut() {
            if *weight >= MAX_WEIGHT - f64::EPSILON {
                *weight = MAX_WEIGHT;
            } else if *weight != 0.0 {
                *weight += need_to_allocate * (*weight / sum_of_weights_to_modif);
            }
        }

        // still some weights can be >= 0;
        // repeat the redistribution process
        if infinite_loop_counter > 2_000 {
            panic!("{weights:#?}\n{scaled_weights:#?}\n{need_to_allocate}\n{sum_of_weights_to_modif}\nIt seems like we have an infinite loop");
        }
    }

    // asserts that sum of weights is 1.0
    assert!((scaled_weights.sum() - 1.0_f64).abs() <= f64::EPSILON * 100., "{weights:#?}\n{}", scaled_weights.sum());
    // asserts that max() value is MAX_WEIGHT
    assert!(scaled_weights.max() <= MAX_WEIGHT + f64::EPSILON * 100., "{}", scaled_weights.max());
    // assert min value != 0 >= MIN_NON_ZERO_WEIGHT
    let min_nonzero = *scaled_weights.iter().filter(|&&weight| weight > f64::EPSILON).min_by(|a, b| a.total_cmp(b)).unwrap();
    assert!(min_nonzero >= MIN_NON_ZERO_WEIGHT - f64::EPSILON * 100., "{}", min_nonzero);

    scaled_weights
}


// Globally available, lazy init at first access:
static HIST_MEAN_COV: LazyLock<(DVector<f64>, DMatrix<f64>)> = LazyLock::new(|| {
    let (mean, cov) = load_mean_cov_matrices(CSV_PATH).expect("failed to load mean/cov");
    (mean, cov)
});

fn main() {
    let handler = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)  // 32 MiB
        .spawn(|| {
            // Optimization Bounds:
            let u = SVector::<f64, N_ASSETS>::from_row_slice(&[0.0; N_ASSETS]); // lower bound for weights
            let v = SVector::<f64, N_ASSETS>::from_row_slice(&[MAX_WEIGHT; N_ASSETS]); // upper bound for weights

            let nsweeps = 1_000;   // maximum number of sweeps
            let nf = 2_000_000; // maximum number of function evaluations

            let local = 100;    // local search level
            let gamma = f64::EPSILON;  // acceptable relative accuracy for local search
            let smax = 2_000; // number of levels used

            let hess = SMatrix::<f64, N_ASSETS, N_ASSETS>::repeat(1.); // sparsity pattern of Hessian

            let (xbest, fbest, _, _, _, _, exitflag) = mcs::<N_ASSETS>(sharpe_ratio, &u, &v, nsweeps, nf, local, gamma, smax, &hess).unwrap();
            println!("Best weights are: {:#?}", modify_weights(&xbest));
            println!("Exit Flag: {exitflag:?}");
            println!("Best sharpe ration is: {fbest}");
        })
        .expect("failed to spawn thread");

    handler.join().unwrap();
}

#[cfg(test)]
mod tests {
    use nalgebra::SVector;
    use super::*;

    #[test]
    fn test_1() {
        let arr = modify_weights(&SVector::from_row_slice(&[1. / N_ASSETS as f64; N_ASSETS]));
        println!("{:?}", arr);
    }

    // #[test]
    // fn test_2() {
    //     let _ = modify_weights(&SVector::from_row_slice(&[0.05; N_ASSETS]));
    // }
}