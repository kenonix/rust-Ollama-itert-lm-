fn main() {
    let target = std::env::var("TARGET").unwrap_or_else(|_| "x86_64-unknown-linux-gnu".to_string());
    println!("cargo:rustc-link-search=native=/home/kenonix/.cache/litert-lm-sys/v0.10.2/{}", target);
    println!("cargo:rustc-link-lib=dylib=GemmaModelConstraintProvider");
    
    // Also tell the binary where to look for dynamic libraries at runtime
    println!("cargo:rustc-link-arg=-Wl,-rpath,/home/kenonix/.cache/litert-lm-sys/v0.10.2/{}", target);
    println!("cargo:rustc-link-arg=-Wl,-rpath,/home/kenonix/.cache/litert-sys/v0.10.2/{}", target);
}
