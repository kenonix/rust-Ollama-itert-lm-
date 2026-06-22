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

// Re-export main types for library users
pub use manager::LitManager;
pub use mcp::LiteRtMcpService;
pub use native_engine::{NativeEngine, NativeCompletionStream, find_model_path};
pub use process::{LitProcess, ProcessPool};
pub use server::{AppState, ChatCompletionRequest, create_router};

// Re-export common types
pub type Result<T> = std::result::Result<T, anyhow::Error>;

/// Helper to load the native LiteRT-LM C library from the standard path.
pub fn load_native_library() -> Result<ffi::LibLiteRtLm> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    
    // Support both .so (Linux) and .dylib (macOS)
    let lib_name = if cfg!(target_os = "macos") {
        "liblitert-lm.dylib"
    } else if cfg!(target_os = "windows") {
        "litert-lm.dll"
    } else {
        "liblitert-lm.so"
    };
    
    let lib_path = home.join(".cache/litert-lm/lib").join(lib_name);
    
    if !lib_path.exists() {
        return Err(anyhow::anyhow!(
            "Native LiteRT-LM shared library not found at: {}\n\
            Please compile it using build_cpp_library.sh on the target machine.",
            lib_path.display()
        ));
    }
    
    unsafe { ffi::LibLiteRtLm::load(&lib_path).map_err(|e| anyhow::anyhow!("Failed to load native library: {}", e)) }
}
