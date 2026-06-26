use std::ffi::{c_void, CString};
use std::fs;
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
                anyhow::bail!(
                    "Failed to create LiteRT-LM engine settings for model: {}",
                    model_path_str
                );
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
                        let _ = tx.blocking_send(Err(anyhow::anyhow!(
                            "Unknown error in LiteRT-LM C++ stream"
                        )));
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
            let decode_res =
                (self.lib.session_run_decode_async)(self.session, raw_callback, tx_ptr);
            if decode_res != 0 {
                // Clean up the leaked box if decode starting failed
                let _ = Box::from_raw(tx_ptr as *mut ChanSender);
                anyhow::bail!(
                    "LiteRT-LM decode stream failed to start with code: {}",
                    decode_res
                );
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

fn collect_candidate_model_paths(model_name: &str, home: &Path) -> Vec<PathBuf> {
    let model_id = model_name.strip_suffix(".litertlm").unwrap_or(model_name);
    let mut candidates = vec![PathBuf::from(model_name)];

    if model_name.ends_with(".litertlm") {
        candidates.push(PathBuf::from(model_id));
    } else {
        candidates.push(PathBuf::from(format!("{model_name}.litertlm")));
    }

    let litert_models = home.join(".litert-lm/models");
    candidates.push(litert_models.join(model_name));
    candidates.push(litert_models.join(format!("{model_id}.litertlm")));

    let cache_dir = home.join(".cache/litert-lm");
    candidates.push(cache_dir.join(model_name));
    candidates.push(cache_dir.join(format!("{model_id}.litertlm")));

    let mais_dir = home.join("mais");
    candidates.push(mais_dir.join(model_name));
    candidates.push(mais_dir.join(format!("{model_id}.litertlm")));

    let home_gits_mais = home.join("gits/mais");
    candidates.push(home_gits_mais.join(model_name));
    candidates.push(home_gits_mais.join(format!("{model_id}.litertlm")));

    if let Ok(current_dir) = std::env::current_dir() {
        for ancestor in current_dir.ancestors() {
            let ancestor_mais = ancestor.join("mais");
            candidates.push(ancestor_mais.join(model_name));
            candidates.push(ancestor_mais.join(format!("{model_id}.litertlm")));
        }
    }

    candidates
}

fn stage_model_for_lit_cli(
    local_model_path: &Path,
    model_name: &str,
    home: &Path,
) -> Option<String> {
    let cache_dir = home.join(".cache/litert-lm");
    if let Err(err) = fs::create_dir_all(&cache_dir) {
        tracing::warn!(path = %cache_dir.display(), error = %err, "Failed to create lit cache directory");
        return None;
    }

    let target_name = if model_name.ends_with(".litertlm") {
        model_name.to_string()
    } else {
        format!("{model_name}.litertlm")
    };

    let staged_path = cache_dir.join(&target_name);
    if !staged_path.exists() {
        if let Err(err) = fs::copy(local_model_path, &staged_path) {
            tracing::warn!(source = %local_model_path.display(), target = %staged_path.display(), error = %err, "Failed to stage local model into lit cache");
            return None;
        }
    }

    Some(target_name)
}

/// Return the best local model path for a provided model identifier.
pub fn find_model_path(model_name: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(model_name);
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?;
    let candidates = collect_candidate_model_paths(model_name, &home);

    for candidate in &candidates {
        if candidate.exists() {
            return Ok(candidate.clone());
        }
    }

    let candidate_list = candidates
        .iter()
        .map(|candidate| format!("- {}", candidate.display()))
        .collect::<Vec<_>>()
        .join("\n");

    anyhow::bail!(
        "Model '{}' not found in candidate paths:\n{}",
        model_name,
        candidate_list
    )
}

/// Resolve a model identifier to the most suitable value for the lit CLI.
pub fn resolve_model_argument(model_name: &str) -> String {
    if let Some(home) = dirs::home_dir() {
        let candidates = collect_candidate_model_paths(model_name, &home);
        if let Some(path) = candidates.iter().find(|candidate| candidate.exists()) {
            if let Some(staged_name) = stage_model_for_lit_cli(path, model_name, &home) {
                return staged_name;
            }
            return path.to_string_lossy().into_owned();
        }
    } else if Path::new(model_name).exists() {
        return model_name.to_string();
    }

    model_name
        .strip_suffix(".litertlm")
        .unwrap_or(model_name)
        .to_string()
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

#[cfg(test)]
mod tests {
    use super::{find_model_path, resolve_model_argument};
    use std::env;
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn find_model_path_recognizes_models_in_home_mais_directory() {
        let temp_home = env::temp_dir().join(format!("litert-lm-test-{}", std::process::id()));
        let model_dir = temp_home.join("mais");
        fs::create_dir_all(&model_dir).unwrap();

        let model_path = model_dir.join("demo-model.litertlm");
        fs::write(&model_path, b"fake").unwrap();

        let previous_home = env::var_os("HOME");
        env::set_var("HOME", &temp_home);

        let result = find_model_path("demo-model.litertlm");

        if let Some(previous_home) = previous_home {
            env::set_var("HOME", previous_home);
        } else {
            env::remove_var("HOME");
        }

        let found_path = result.expect("expected the model to be discovered");
        assert_eq!(found_path, PathBuf::from(&model_path));

        let _ = fs::remove_dir_all(&temp_home);
    }

    #[test]
    fn resolve_model_argument_stages_local_models_into_lit_cache() {
        let temp_home =
            env::temp_dir().join(format!("litert-lm-test-cache-{}", std::process::id()));
        let model_dir = temp_home.join("gits/mais");
        fs::create_dir_all(&model_dir).unwrap();

        let model_path = model_dir.join("demo-model.litertlm");
        fs::write(&model_path, b"fake").unwrap();

        let previous_home = env::var_os("HOME");
        env::set_var("HOME", &temp_home);

        let resolved = resolve_model_argument("demo-model.litertlm");

        if let Some(previous_home) = previous_home {
            env::set_var("HOME", previous_home);
        } else {
            env::remove_var("HOME");
        }

        assert_eq!(resolved, "demo-model.litertlm");
        assert!(temp_home
            .join(".cache/litert-lm/demo-model.litertlm")
            .exists());

        let _ = fs::remove_dir_all(&temp_home);
    }
}
