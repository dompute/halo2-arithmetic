use std::env;

fn main() {
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();
    println!("cargo:warning=CUDA Status:{}", cuda_enabled);
    if !cuda_enabled {
        return;
    }

    let profile = match env::var("PROFILE").unwrap().as_str() {
        "release" => "Release",
        _ => "Debug",
    };

    cc::Build::new()
        .cuda(true)
        .opt_level(3)
        .flag("-cudart=shared")
        .file("libfam/src/base.cu")
        .file("libfam/src/stub.cu")
        .file("libfam/src/arithmetic/msm.cu")
        .compile("libfam.a");

    // let dst = Config::new("libfam")
    //     .define("CMAKE_BUILD_TYPE", profile)
    //     .define("DG_TEST", "OFF")
    //     .build();
    // println!("cargo:rustc-link-search=native={}/lib", dst.display());
    // println!("cargo:rustc-link-lib=static=fam");
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64/stubs");
    println!("cargo:rustc-link-lib=cuda");
}
