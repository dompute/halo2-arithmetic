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

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .opt_level(3)
        .cudart("static")
        .file("libfam/src/base.cu")
        .file("libfam/src/stub.cu")
        .file("libfam/src/arithmetic/msm.cu");
    println!("cargo:warning={:?}", build);
    build.compile("libfam.a");

    println!("cargo:rustc-link-search=native=/opt/cuda/lib64/stubs");
    println!("cargo:rustc-link-lib=cuda");

    println!("cargo:rerun-if-changed=libfam/src/base.cu");
    println!("cargo:rerun-if-changed=libfam/src/stub.cu");
    println!("cargo:rerun-if-changed=libfam/src/arithmetic/msm.cu");
}
