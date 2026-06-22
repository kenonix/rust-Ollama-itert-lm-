use std::ffi::{CString, c_void};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;

use crate::ffi::{self, LibLiteRtLm, LiteRtLmEngine, LiteRtLmSession};

pub struct NativeEngine {
    lib: Arc<LibLiteRtLm>,
    engine: *mut LiteRtLmEngine,
    session: *mut LiteRtLmSession,
}

// Thread-safety markers
unsafe impl Send for NativeEngine {}
unsafe impl Sync for NativeEngine {}

impl NativeEngine {
    /// Create a new NativeEngine instance for a model at `model_path` and target `backend`.
    pub fn new(lib: Arc<LibLiteRtLm>, model_path: &Path, backend: &str) -> anyhow::Result<Self> {
        let model_path_str = model_path.to_string_lossy();
        let model_path_cstr = CString::new(model_path_str.as_ref())?;
        let backend_cstr = CString::new(backend)?;

        unsafe {
            // Set min log severity (2 is INFO)
            (lib.set_min_log_level)(2);

            let settings = (lib.engine_settings_create)(
                model_path_cstr.as_ptr(),
                backend_cstr.as_ptr(),
                std::ptr::null(), // vision_backend
                std::ptr::null(), // audio_backend
            );
            if settings.is_null() {
                anyhow::bail!("Failed to create LiteRT-LM engine settings for model: {}", model_path_str);
            }

            // Default max tokens
            (lib.engine_settings_set_max_num_tokens)(settings, 2048);

            // Create engine
            let engine = (lib.engine_create)(settings);
            (lib.engine_settings_delete)(settings);

            if engine.is_null() {
                anyhow::bail!("Failed to create LiteRT-LM engine instance");
            }

            // Create session config
            let config = (lib.session_config_create)();
            if config.is_null() {
                (lib.engine_delete)(engine);
                anyhow::bail!("Failed to create session config");
            }

            (lib.session_config_set_max_output_tokens)(config, 2048);

            // Create session
            let session = (lib.engine_create_session)(engine, config);
            (lib.session_config_delete)(config);

            if session.is_null() {
                (lib.engine_delete)(engine);
                anyhow::bail!("Failed to create LiteRT-LM session");
            }

            Ok(Self {
                lib,
                engine,
                session,
            })
        }
    }

    /// Run inference for `prompt` and stream the responses.
    pub fn run_completion_stream(&self, prompt: &str) -> anyhow::Result<NativeCompletionStream> {
        let prompt_cstr = CString::new(prompt)?;

        unsafe {
            // 1. Run Prefill (blocking call)
            let input_data = ffi::LiteRtLmInputData {
                r#type: ffi::LiteRtLmInputDataType::Text,
                data: prompt_cstr.as_ptr() as *const c_void,
                size: prompt.len(),
            };

            let prefill_res = (self.lib.session_run_prefill)(self.session, &input_data, 1);
            if prefill_res != 0 {
                anyhow::bail!("LiteRT-LM prefill failed with exit code: {}", prefill_res);
            }

            // 2. Setup communication channel
            let (tx, rx) = mpsc::channel::<anyhow::Result<String>>(100);

            // We leak the sender so the C++ background thread can safely receive it as user_data pointer.
            // The callback MUST reclaim this pointer when the stream ends/errors.
            let tx_boxed = Box::new(tx);
            let tx_ptr = Box::into_raw(tx_boxed) as *mut c_void;

            type ChanSender = mpsc::Sender<anyhow::Result<String>>;

            unsafe extern "C" fn raw_callback(
                callback_data: *mut c_void,
                chunk: *const std::ffi::c_char,
                is_final: bool,
                error_msg: *const std::ffi::c_char,
            ) {
                if callback_data.is_null() {
                    return;
                }

                let tx = &*(callback_data as *const ChanSender);

                if !error_msg.is_null() {
                    if let Ok(err_cstr) = std::ffi::CStr::from_ptr(error_msg).to_str() {
                        let _ = tx.blocking_send(Err(anyhow::anyhow!("{}", err_cstr)));
                    } else {
                        let _ = tx.blocking_send(Err(anyhow::anyhow!("Unknown error in LiteRT-LM C++ stream")));
                    }
                    // Reclaim ownership to free the channel sender Box
                    let _ = Box::from_raw(callback_data as *mut ChanSender);
                    return;
                }

                if !chunk.is_null() {
                    if let Ok(chunk_str) = std::ffi::CStr::from_ptr(chunk).to_str() {
                        let _ = tx.blocking_send(Ok(chunk_str.to_string()));
                    }
                }

                if is_final {
                    // Reclaim ownership to free the channel sender Box
                    let _ = Box::from_raw(callback_data as *mut ChanSender);
                }
            }

            // 3. Start asynchronous decode
            let decode_res = (self.lib.session_run_decode_async)(self.session, raw_callback, tx_ptr);
            if decode_res != 0 {
                // Clean up the leaked box if decode starting failed
                let _ = Box::from_raw(tx_ptr as *mut ChanSender);
                anyhow::bail!("LiteRT-LM decode stream failed to start with code: {}", decode_res);
            }

            Ok(NativeCompletionStream {
                lib: self.lib.clone(),
                session: self.session,
                stream: ReceiverStream::new(rx),
            })
        }
    }
}

impl Drop for NativeEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.session.is_null() {
                (self.lib.session_delete)(self.session);
            }
            if !self.engine.is_null() {
                (self.lib.engine_delete)(self.engine);
            }
        }
    }
}

pub struct NativeCompletionStream {
    lib: Arc<LibLiteRtLm>,
    session: *mut LiteRtLmSession,
    stream: ReceiverStream<anyhow::Result<String>>,
}

unsafe impl Send for NativeCompletionStream {}
unsafe impl Sync for NativeCompletionStream {}

impl Stream for NativeCompletionStream {
    type Item = anyhow::Result<String>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::pin::Pin::new(&mut self.stream).poll_next(cx)
    }
}

impl Drop for NativeCompletionStream {
    fn drop(&mut self) {
        // Cancel C++ execution if the stream is dropped prematurely
        unsafe {
            (self.lib.session_cancel_process)(self.session);
        }
    }
}

/// Utility function to locate the model path.
pub fn find_model_path(model_name: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(model_name);
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    
    // Check ~/.litert-lm/models/
    let litert_models = home.join(".litert-lm/models");
    let candidate1 = litert_models.join(model_name);
    if candidate1.exists() {
        return Ok(candidate1);
    }
    
    let candidate2 = litert_models.join(format!("{}.litertlm", model_name));
    if candidate2.exists() {
        return Ok(candidate2);
    }

    // Check ~/.cache/litert-lm/
    let cache_dir = home.join(".cache/litert-lm");
    let candidate3 = cache_dir.join(model_name);
    if candidate3.exists() {
        return Ok(candidate3);
    }

    let candidate4 = cache_dir.join(format!("{}.litertlm", model_name));
    if candidate4.exists() {
        return Ok(candidate4);
    }

    anyhow::bail!(
        "Model '{}' not found in candidate paths:\n\
         - {}\n\
         - {}\n\
         - {}\n\
         - {}",
        model_name,
        candidate1.display(),
        candidate2.display(),
        candidate3.display(),
        candidate4.display()
    )
}

impl std::fmt::Debug for NativeEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeEngine")
            .field("engine", &self.engine)
            .field("session", &self.session)
            .finish()
    }
}

impl std::fmt::Debug for NativeCompletionStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeCompletionStream")
            .field("session", &self.session)
            .finish()
    }
}

