//! LiteRT-LM: A Rust library for LiteRT-LM model inference
//!
//! This library provides:
//! - Model download and management
//! - Process pool management for efficient inference
//! - Streaming completions
//! - MCP (Model Context Protocol) service
//! - OpenAI-compatible API server
//!
//! # Example
//!
//! ```no_run
//! use litert_lm::{LitManager, Result};
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     let manager = LitManager::new().await?;
//!
//!     // Pull a model
//!     manager.pull("gemma-3n-E4B", None, None).await?;
//!
//!     // Run completion
//!     let response = manager.run_completion("gemma-3n-E4B", "Hello!").await?;
//!     println!("{}", response);
//!
//!     Ok(())
//! }
//! ```

pub mod binary;
pub mod ffi;
pub mod manager;
pub mod mcp;
pub mod native_engine;
pub mod process;
pub mod server;

use std::path::{Path, PathBuf};
use std::process::Command;

// Re-export main types for library users
pub use manager::LitManager;
pub use mcp::LiteRtMcpService;
pub use native_engine::{find_model_path, NativeCompletionStream, NativeEngine};
pub use process::{LitProcess, ProcessPool};
pub use server::{create_router, AppState, ChatCompletionRequest};

// Re-export common types
pub type Result<T> = std::result::Result<T, anyhow::Error>;

fn native_library_path(home: &Path) -> PathBuf {
    let lib_name = if cfg!(target_os = "macos") {
        "liblitert-lm.dylib"
    } else if cfg!(target_os = "windows") {
        "litert-lm.dll"
    } else {
        "liblitert-lm.so"
    };

    home.join(".cache/litert-lm/lib").join(lib_name)
}

fn find_build_script() -> Option<(PathBuf, PathBuf)> {
    let script_name = "build_cpp_library.sh";

    // 1. Try current working directory
    if let Ok(cwd) = std::env::current_dir() {
        let script = cwd.join(script_name);
        if script.exists() {
            return Some((script, cwd));
        }
    }

    // 2. Try current executable directory
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // Check exe_dir
            let script = exe_dir.join(script_name);
            if script.exists() {
                return Some((script, exe_dir.to_path_buf()));
            }
            // Check exe_dir/..
            if let Some(parent) = exe_dir.parent() {
                let script = parent.join(script_name);
                if script.exists() {
                    return Some((script, parent.to_path_buf()));
                }
                // Check exe_dir/../..
                if let Some(grandparent) = parent.parent() {
                    let script = grandparent.join(script_name);
                    if script.exists() {
                        return Some((script, grandparent.to_path_buf()));
                    }
                }
            }
        }
    }

    // 3. Try compile-time manifest parent
    let manifest_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let script = manifest_root.join(script_name);
    if script.exists() {
        return Some((script, manifest_root.to_path_buf()));
    }

    None
}

fn ensure_native_library_available() -> Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    let lib_path = native_library_path(&home);

    if lib_path.exists() {
        return Ok(lib_path);
    }

    let (build_script, build_cwd) = find_build_script().ok_or_else(|| {
        anyhow::anyhow!(
            "Native LiteRT-LM shared library not found at {}, and build_cpp_library.sh was not found in CWD, executable directory, or manifest parent.",
            lib_path.display()
        )
    })?;

    tracing::info!(
        path = %lib_path.display(),
        script = %build_script.display(),
        "Native LiteRT-LM shared library is missing; building it now"
    );

    let status = Command::new("bash")
        .arg(&build_script)
        .current_dir(&build_cwd)
        .status()
        .map_err(|e| anyhow::anyhow!("Failed to start native library build: {e}"))?;

    if !status.success() {
        return Err(anyhow::anyhow!(
            "Failed to build LiteRT-LM native library with {} (exit code: {:?})",
            build_script.display(),
            status.code()
        ));
    }

    if !lib_path.exists() {
        return Err(anyhow::anyhow!(
            "Native LiteRT-LM shared library still missing after build at {}",
            lib_path.display()
        ));
    }

    Ok(lib_path)
}

/// Helper to load the native LiteRT-LM C library from the standard path.
pub fn load_native_library() -> Result<ffi::LibLiteRtLm> {
    let lib_path = ensure_native_library_available()?;

    if cfg!(target_os = "linux") {
        if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
            let lib_dir = lib_path.parent().unwrap_or_else(|| Path::new("."));
            let new_value = format!("{}:{}", lib_dir.display(), existing.to_string_lossy());
            std::env::set_var("LD_LIBRARY_PATH", new_value);
        } else {
            let lib_dir = lib_path.parent().unwrap_or_else(|| Path::new("."));
            std::env::set_var("LD_LIBRARY_PATH", lib_dir);
        }
    }

    unsafe {
        match ffi::LibLiteRtLm::load(&lib_path) {
            Ok(lib) => Ok(lib),
            Err(e) => {
                // If it exists but failed to load, it might be compiled for a different architecture
                // or corrupted. Delete it and attempt to rebuild once.
                tracing::warn!(
                    path = %lib_path.display(),
                    error = %e,
                    "Failed to load existing shared library. Attempting to delete and rebuild..."
                );
                
                let _ = std::fs::remove_file(&lib_path);
                
                // Try ensuring library availability again (which will force a rebuild now that it doesn't exist)
                let lib_path = ensure_native_library_available()?;
                
                ffi::LibLiteRtLm::load(&lib_path).map_err(|err| {
                    anyhow::anyhow!(
                        "Failed to load native library after rebuild from {}: {}",
                        lib_path.display(),
                        err
                    )
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::native_library_path;
    use std::path::Path;

    #[test]
    fn native_library_path_uses_cache_directory() {
        let home = Path::new("/tmp/test-home");
        let path = native_library_path(home);
        assert!(path.ends_with(".cache/litert-lm/lib/liblitert-lm.so"));
    }
}
