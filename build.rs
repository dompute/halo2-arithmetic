use cmake::Config;
use std::env;

fn main() {
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    if !cuda_enabled {
        return;
    }

    let profile = match env::var("PROFILE").unwrap().as_str() {
        "release" => "Release",
        _ => "Debug",
    };

    let dst = Config::new("libfam")
        .define("CMAKE_BUILD_TYPE", profile)
        .define("DG_TEST", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=foo");
}
