use candle_core::{bail, Result, Tensor};

pub fn fit(xs: &Tensor) -> Result<Vec<f32>> {
    // Scale to unit variance
    let mu = xs.mean(0)?;
    let x_centered = xs.broadcast_sub(&mu)?;

    bail!("Not implemented")
}

pub fn zipf(xs: &Tensor) -> Result<Tensor> {
    let weighting = Tensor::arange::<u32>(1, (xs.dim(0)? + 1) as u32, xs.device())?.log()?;
    xs.broadcast_mul(&weighting)
}
