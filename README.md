## Halo2-arithmetic

The repo contains basic arithmetic operations implemented in Halo2 and is considered as a cuda-powered replacement of the MSM , FFT and graph evaluation in the original Halo2.

## How to use it

We port some strutures and functions from halo2. You should replace those for your own halo2 build. Includes:

- best_msm
- best_fft
- GraphEvaluator::evaluate

## How to enable cuda

``` toml
[dependencies]
halo2-arithmetic = { git = "https://github.com/dompute/halo2-arithmetic", features = ["cuda"] }

```
