use std::env;

fn main() {
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    println!("cargo:warning=CUDA Status:{}", cuda_enabled);
    if !cuda_enabled {
        return;
    }

    // let profile = match env::var("PROFILE").unwrap().as_str() {
    //     "release" => "Release",
    //     _ => "Debug",
    // };

    cc::Build::new()
        .cuda(true)
        .opt_level(3)
        .cudart("static")
        .file("libfam/src/stub.cu")
        .compile("libfam.a");

    println!("cargo:rerun-if-changed=libfam/src/arithmetic.cuh");
    println!("cargo:rerun-if-changed=libfam/src/fft.cuh");
    println!("cargo:rerun-if-changed=libfam/src/ff.cuh");
    println!("cargo:rerun-if-changed=libfam/src/bn256.cuh");
    println!("cargo:rerun-if-changed=libfam/src/pippenger.cuh");
    println!("cargo:rerun-if-changed=libfam/src/graph.cuh");
    println!("cargo:rerun-if-changed=libfam/src/stub.cu");
}
