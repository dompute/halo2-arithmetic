#![feature(min_specialization)]

#[cfg(feature = "cuda")]
mod stub;

pub mod arithmetic;
pub mod graph;

pub mod value_source;

#[cfg(feature = "cuda")]
pub use stub::{fft_fr, msm_fr_g1, U256};
