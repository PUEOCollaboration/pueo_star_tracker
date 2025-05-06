use prost_build;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut config = prost_build::Config::new();
    config.protoc_arg("--experimental_allow_proto3_optional");

    tonic_build::configure().compile_with_config(
        config,
        &["src/proto/cedar_detect.proto"], 
        &["src/proto"])?;

    // Optional: Re-run build if proto changes
    println!("cargo:rerun-if-changed=src/proto/cedar_detect.proto");

    Ok(())
}
