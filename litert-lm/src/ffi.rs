use std::ffi::{c_char, c_int, c_void};
use libloading::Library;

#[repr(C)]
pub struct LiteRtLmEngineSettings {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmEngine {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmSessionConfig {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LiteRtLmSession {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum LiteRtLmInputDataType {
    Text = 0,
    Image = 1,
    ImageEnd = 2,
    Audio = 3,
    AudioEnd = 4,
}

#[repr(C)]
pub struct LiteRtLmInputData {
    pub r#type: LiteRtLmInputDataType,
    pub data: *const c_void,
    pub size: usize,
}

pub type LiteRtLmStreamCallback = unsafe extern "C" fn(
    callback_data: *mut c_void,
    chunk: *const c_char,
    is_final: bool,
    error_msg: *const c_char,
);

pub struct LibLiteRtLm {
    _lib: Library,
    pub set_min_log_level: unsafe extern "C" fn(level: c_int),
    pub engine_settings_create: unsafe extern "C" fn(
        model_path: *const c_char,
        backend_str: *const c_char,
        vision_backend_str: *const c_char,
        audio_backend_str: *const c_char,
    ) -> *mut LiteRtLmEngineSettings,
    pub engine_settings_delete: unsafe extern "C" fn(settings: *mut LiteRtLmEngineSettings),
    pub engine_settings_set_max_num_tokens: unsafe extern "C" fn(
        settings: *mut LiteRtLmEngineSettings,
        max_num_tokens: c_int,
    ),
    pub engine_settings_set_num_threads: unsafe extern "C" fn(
        settings: *mut LiteRtLmEngineSettings,
        num_threads: c_int,
    ),
    pub engine_create: unsafe extern "C" fn(settings: *const LiteRtLmEngineSettings) -> *mut LiteRtLmEngine,
    pub engine_delete: unsafe extern "C" fn(engine: *mut LiteRtLmEngine),
    pub session_config_create: unsafe extern "C" fn() -> *mut LiteRtLmSessionConfig,
    pub session_config_set_max_output_tokens: unsafe extern "C" fn(
        config: *mut LiteRtLmSessionConfig,
        max_output_tokens: c_int,
    ),
    pub session_config_delete: unsafe extern "C" fn(config: *mut LiteRtLmSessionConfig),
    pub engine_create_session: unsafe extern "C" fn(
        engine: *mut LiteRtLmEngine,
        config: *mut LiteRtLmSessionConfig,
    ) -> *mut LiteRtLmSession,
    pub session_delete: unsafe extern "C" fn(session: *mut LiteRtLmSession),
    pub session_run_prefill: unsafe extern "C" fn(
        session: *mut LiteRtLmSession,
        inputs: *const LiteRtLmInputData,
        num_inputs: usize,
    ) -> c_int,
    pub session_run_decode_async: unsafe extern "C" fn(
        session: *mut LiteRtLmSession,
        callback: LiteRtLmStreamCallback,
        callback_data: *mut c_void,
    ) -> c_int,
    pub session_cancel_process: unsafe extern "C" fn(session: *mut LiteRtLmSession),
}

impl LibLiteRtLm {
    pub unsafe fn load(path: &std::path::Path) -> Result<Self, libloading::Error> {
        let lib = Library::new(path)?;

        let set_min_log_level = *lib.get(b"litert_lm_set_min_log_level")?;
        let engine_settings_create = *lib.get(b"litert_lm_engine_settings_create")?;
        let engine_settings_delete = *lib.get(b"litert_lm_engine_settings_delete")?;
        let engine_settings_set_max_num_tokens = *lib.get(b"litert_lm_engine_settings_set_max_num_tokens")?;
        let engine_settings_set_num_threads = *lib.get(b"litert_lm_engine_settings_set_num_threads")?;
        let engine_create = *lib.get(b"litert_lm_engine_create")?;
        let engine_delete = *lib.get(b"litert_lm_engine_delete")?;
        let session_config_create = *lib.get(b"litert_lm_session_config_create")?;
        let session_config_set_max_output_tokens = *lib.get(b"litert_lm_session_config_set_max_output_tokens")?;
        let session_config_delete = *lib.get(b"litert_lm_session_config_delete")?;
        let engine_create_session = *lib.get(b"litert_lm_engine_create_session")?;
        let session_delete = *lib.get(b"litert_lm_session_delete")?;
        let session_run_prefill = *lib.get(b"litert_lm_session_run_prefill")?;
        let session_run_decode_async = *lib.get(b"litert_lm_session_run_decode_async")?;
        let session_cancel_process = *lib.get(b"litert_lm_session_cancel_process")?;

        Ok(Self {
            _lib: lib,
            set_min_log_level,
            engine_settings_create,
            engine_settings_delete,
            engine_settings_set_max_num_tokens,
            engine_settings_set_num_threads,
            engine_create,
            engine_delete,
            session_config_create,
            session_config_set_max_output_tokens,
            session_config_delete,
            engine_create_session,
            session_delete,
            session_run_prefill,
            session_run_decode_async,
            session_cancel_process,
        })
    }
}
