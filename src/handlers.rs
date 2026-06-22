use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde;
use serde_json;
use tokio::sync::mpsc;
use crate::AppState;
use crate::agentic::{run_agentic_loop, ServerStreamEvent};
use crate::utils::get_iso8601_now;

#[derive(serde::Deserialize, Debug)]
pub struct ChatRequest {
    pub messages: Option<Vec<serde_json::Value>>,
    pub stream: Option<bool>,
    pub options: Option<serde_json::Value>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub max_tokens: Option<i64>,
    pub max_output_tokens: Option<i64>,
    pub num_predict: Option<i64>,
    pub tools: Option<Vec<serde_json::Value>>,
}

#[derive(serde::Deserialize, Debug)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    pub template: Option<String>,
    pub stream: Option<bool>,
    pub options: Option<serde_json::Value>,
    pub keep_alive: Option<serde_json::Value>,
}

#[derive(serde::Deserialize, Debug)]
pub struct EmbedRequest {
    pub model: String,
    pub input: serde_json::Value, // Can be string or array of strings
    pub truncate: Option<bool>,
    pub options: Option<serde_json::Value>,
    pub keep_alive: Option<serde_json::Value>,
}

#[derive(serde::Deserialize, Debug)]
pub struct CreateRequest {
    pub model: String,
    pub path: Option<String>,
    pub modelfile: Option<String>,
    pub stream: Option<bool>,
}

#[derive(serde::Deserialize, Debug)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

#[derive(serde::Deserialize, Debug)]
pub struct PullRequest {
    pub model: String,
    pub insecure: Option<bool>,
    pub stream: Option<bool>,
}

#[derive(serde::Deserialize, Debug)]
pub struct PushRequest {
    pub model: String,
    pub insecure: Option<bool>,
    pub stream: Option<bool>,
}

#[derive(serde::Deserialize, Debug)]
pub struct DeleteRequest {
    pub model: String,
}

#[derive(serde::Deserialize, Debug)]
pub struct ShowRequest {
    pub model: String,
    pub verbose: Option<bool>,
}

pub async fn handle_root() -> &'static str {
    "Ollama is running"
}

pub async fn handle_version() -> impl IntoResponse {
    Json(serde_json::json!({
        "version": "0.1.48"
    }))
}

pub async fn handle_tags(State(state): State<AppState>) -> impl IntoResponse {
    let model_entry = serde_json::json!({
        "name": state.model_name,
        "model": state.model_name,
        "modified_at": get_iso8601_now(),
        "size": 2500000000i64,
        "digest": "0000000000000000000000000000000000000000000000000000000000000000",
        "details": {
            "format": "tflite",
            "family": "litert",
            "families": ["litert"],
            "parameter_size": "2B",
            "quantization_level": "none"
        }
    });
    let response = serde_json::json!({
        "models": vec![model_entry]
    });
    Json(response)
}

pub async fn handle_ps(State(state): State<AppState>) -> impl IntoResponse {
    let model_entry = serde_json::json!({
        "name": state.model_name,
        "model": state.model_name,
        "size": 2500000000i64,
        "digest": "0000000000000000000000000000000000000000000000000000000000000000",
        "details": {
            "format": "tflite",
            "family": "litert",
            "families": ["litert"],
            "parameter_size": "2B",
            "quantization_level": "none"
        },
        "expires_at": "0001-01-01T00:00:00Z",
        "size_vram": 2500000000i64
    });
    let response = serde_json::json!({
        "models": vec![model_entry]
    });
    Json(response)
}

pub async fn handle_models(State(state): State<AppState>) -> impl IntoResponse {
    let model_entry = serde_json::json!({
        "id": state.model_name,
        "object": "model",
        "created": chrono::Utc::now().timestamp(),
        "owned_by": "litert"
    });
    let response = serde_json::json!({
        "object": "list",
        "data": vec![model_entry]
    });
    Json(response)
}

pub async fn handle_chat(
    State(state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: ChatRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let want_stream = req.stream.unwrap_or(true);
    let mut sys_msg = state.system_prompt.clone();
    let mut history_arr = serde_json::json!([]);
    let mut current_msg_j = serde_json::json!({
        "role": "user",
        "content": ""
    });
    
    if let Some(messages) = &req.messages {
        if !messages.is_empty() {
            current_msg_j = messages.last().unwrap().clone();
            let len = messages.len();
            for i in 0..len - 1 {
                let msg = &messages[i];
                if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
                    if let Some(c) = msg.get("content").and_then(|c| c.as_str()) {
                        sys_msg = c.to_string();
                    }
                } else {
                    history_arr.as_array_mut().unwrap().push(msg.clone());
                }
            }
        }
    }
    
    let mut opt = serde_json::json!({
        "max_output_tokens": 262144,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    });
    if let Some(ref req_opt) = req.options {
        if let Some(t) = req_opt.get("temperature") { opt["temperature"] = t.clone(); }
        if let Some(p) = req_opt.get("top_p") { opt["top_p"] = p.clone(); }
        if let Some(k) = req_opt.get("top_k") { opt["top_k"] = k.clone(); }
        if let Some(m) = req_opt.get("max_output_tokens") { opt["max_output_tokens"] = m.clone(); }
        if let Some(m) = req_opt.get("max_tokens") { opt["max_output_tokens"] = m.clone(); }
        if let Some(m) = req_opt.get("num_predict") { opt["max_output_tokens"] = m.clone(); }
    } else {
        if let Some(t) = req.temperature { opt["temperature"] = serde_json::json!(t); }
        if let Some(p) = req.top_p { opt["top_p"] = serde_json::json!(p); }
        if let Some(k) = req.top_k { opt["top_k"] = serde_json::json!(k); }
        if let Some(m) = req.max_tokens { opt["max_output_tokens"] = serde_json::json!(m); }
        if let Some(m) = req.max_output_tokens { opt["max_output_tokens"] = serde_json::json!(m); }
        if let Some(m) = req.num_predict { opt["max_output_tokens"] = serde_json::json!(m); }
    }
    
    let request_tools = req.tools.clone().map(|t| serde_json::Value::Array(t));
    let history_json_str = history_arr.to_string();
    let current_msg_str = current_msg_j.to_string();
    let config_json_str = opt.to_string();
    let manager = state.manager.clone();
    let model_name = state.model_name.clone();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<ServerStreamEvent>();
    
    tokio::spawn(async move {
        run_agentic_loop(
            manager,
            model_name,
            sys_msg,
            history_json_str,
            current_msg_str,
            Some(config_json_str),
            request_tools,
            event_tx,
        ).await;
    });
    
    if want_stream {
        let model_name = state.model_name.clone();
        let stream = async_stream::stream! {
            let mut last_tool_calls: Option<serde_json::Value> = None;
            while let Some(event) = event_rx.recv().await {
                match event {
                    ServerStreamEvent::Chunk(c) => {
                        let chunk_j = serde_json::json!({
                            "model": model_name,
                            "created_at": get_iso8601_now(),
                            "message": {
                                "role": "assistant",
                                "content": c
                            },
                            "done": false
                        });
                        yield Ok::<_, anyhow::Error>(format!("{}\n", chunk_j.to_string()));
                    }
                    ServerStreamEvent::ToolCall { raw_tool_calls_json } => {
                        if let Ok(j) = serde_json::from_str::<serde_json::Value>(&raw_tool_calls_json) {
                            if let Some(tc) = j.get("tool_calls") {
                                last_tool_calls = Some(tc.clone());
                                let chunk_j = serde_json::json!({
                                    "model": model_name,
                                    "created_at": get_iso8601_now(),
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": tc
                                    },
                                    "done": false
                                });
                                yield Ok::<_, anyhow::Error>(format!("{}\n", chunk_j.to_string()));
                            }
                        }
                    }
                    ServerStreamEvent::ToolResult { name, result } => {
                        let chunk_j = serde_json::json!({
                            "model": model_name,
                            "created_at": get_iso8601_now(),
                            "tool_result": {
                                "name": name,
                                "result": result
                            },
                            "done": false
                        });
                        yield Ok::<_, anyhow::Error>(format!("{}\n", chunk_j.to_string()));
                    }
                    ServerStreamEvent::Guidance(g) => {
                        let chunk_j = serde_json::json!({
                            "model": model_name,
                            "created_at": get_iso8601_now(),
                            "guidance": g,
                            "done": false
                        });
                        yield Ok::<_, anyhow::Error>(format!("{}\n", chunk_j.to_string()));
                    }
                    ServerStreamEvent::Error(e) => {
                        yield Err(anyhow::anyhow!(e));
                    }
                    ServerStreamEvent::Done { final_history, prompt_tokens, completion_tokens } => {
                        let mut final_j = serde_json::json!({
                            "model": model_name,
                            "done": true,
                            "history": final_history,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens
                        });
                        if let Some(tc) = last_tool_calls.take() {
                            final_j["message"] = serde_json::json!({
                                "role": "assistant",
                                "content": "",
                                "tool_calls": tc
                            });
                        }
                        yield Ok::<_, anyhow::Error>(format!("{}\n", final_j.to_string()));
                    }
                }
            }
        };
        Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let mut final_text = String::new();
        let mut last_history: Vec<serde_json::Value> = Vec::new();
        let mut last_tool_calls: Option<serde_json::Value> = None;
        
        while let Some(event) = event_rx.recv().await {
            match event {
                ServerStreamEvent::Chunk(c) => {
                    final_text.push_str(&c);
                }
                ServerStreamEvent::ToolCall { raw_tool_calls_json } => {
                    if let Ok(j) = serde_json::from_str::<serde_json::Value>(&raw_tool_calls_json) {
                        if let Some(tc) = j.get("tool_calls") {
                            last_tool_calls = Some(tc.clone());
                        }
                    }
                }
                ServerStreamEvent::ToolResult { .. } => {}
                ServerStreamEvent::Guidance(..) => {}
                ServerStreamEvent::Error(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
                }
                ServerStreamEvent::Done { final_history, .. } => {
                    last_history = final_history;
                }
            }
        }
        
        let mut res_j = serde_json::json!({
            "model": state.model_name,
            "message": {
                "role": "assistant",
                "content": final_text
            },
            "done": true,
            "history": last_history
        });
        if let Some(tc) = last_tool_calls {
            res_j["message"] = serde_json::json!({
                "role": "assistant",
                "content": "",
                "tool_calls": tc
            });
        }
        Json(res_j).into_response()
    }
}

pub async fn handle_generate(
    State(state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: GenerateRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let want_stream = req.stream.unwrap_or(true);
    let sys_msg = req.system.unwrap_or_else(|| state.system_prompt.clone());
    let history_arr = serde_json::json!([]);
    let current_msg_j = serde_json::json!({
        "role": "user",
        "content": req.prompt
    });
    
    let mut opt = serde_json::json!({
        "max_output_tokens": 262144,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    });
    if let Some(ref req_opt) = req.options {
        if let Some(t) = req_opt.get("temperature") { opt["temperature"] = t.clone(); }
        if let Some(p) = req_opt.get("top_p") { opt["top_p"] = p.clone(); }
        if let Some(k) = req_opt.get("top_k") { opt["top_k"] = k.clone(); }
        if let Some(m) = req_opt.get("max_output_tokens") { opt["max_output_tokens"] = m.clone(); }
        if let Some(m) = req_opt.get("max_tokens") { opt["max_output_tokens"] = m.clone(); }
        if let Some(m) = req_opt.get("num_predict") { opt["max_output_tokens"] = m.clone(); }
    }
    
    let history_json_str = history_arr.to_string();
    let current_msg_str = current_msg_j.to_string();
    let config_json_str = opt.to_string();
    let manager = state.manager.clone();
    let model_name = state.model_name.clone();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<ServerStreamEvent>();
    
    tokio::spawn(async move {
        run_agentic_loop(
            manager,
            model_name,
            sys_msg,
            history_json_str,
            current_msg_str,
            Some(config_json_str),
            None,
            event_tx,
        ).await;
    });
    
    if want_stream {
        let model_name = state.model_name.clone();
        let stream = async_stream::stream! {
            while let Some(event) = event_rx.recv().await {
                match event {
                    ServerStreamEvent::Chunk(c) => {
                        let chunk_j = serde_json::json!({
                            "model": model_name,
                            "created_at": get_iso8601_now(),
                            "response": c,
                            "done": false
                        });
                        yield Ok::<_, anyhow::Error>(format!("{}\n", chunk_j.to_string()));
                    }
                    ServerStreamEvent::ToolCall { .. } => {}
                    ServerStreamEvent::ToolResult { .. } => {}
                    ServerStreamEvent::Guidance(..) => {}
                    ServerStreamEvent::Error(e) => {
                        yield Err(anyhow::anyhow!(e));
                    }
                    ServerStreamEvent::Done { prompt_tokens, completion_tokens, .. } => {
                        let final_j = serde_json::json!({
                            "model": model_name,
                            "created_at": get_iso8601_now(),
                            "response": "",
                            "done": true,
                            "context": vec![1, 2, 3],
                            "prompt_eval_count": prompt_tokens,
                            "eval_count": completion_tokens
                        });
                        yield Ok::<_, anyhow::Error>(format!("{}\n", final_j.to_string()));
                    }
                }
            }
        };
        Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let mut final_text = String::new();
        let mut prompt_tokens_val = 0;
        let mut completion_tokens_val = 0;
        
        while let Some(event) = event_rx.recv().await {
            match event {
                ServerStreamEvent::Chunk(c) => {
                    final_text.push_str(&c);
                }
                ServerStreamEvent::ToolCall { .. } => {}
                ServerStreamEvent::ToolResult { .. } => {}
                ServerStreamEvent::Guidance(..) => {}
                ServerStreamEvent::Error(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
                }
                ServerStreamEvent::Done { prompt_tokens, completion_tokens, .. } => {
                    prompt_tokens_val = prompt_tokens;
                    completion_tokens_val = completion_tokens;
                }
            }
        }
        
        let res_j = serde_json::json!({
            "model": state.model_name,
            "created_at": get_iso8601_now(),
            "response": final_text,
            "done": true,
            "context": vec![1, 2, 3],
            "prompt_eval_count": prompt_tokens_val,
            "eval_count": completion_tokens_val
        });
        Json(res_j).into_response()
    }
}

pub async fn handle_embed(
    State(state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: EmbedRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let make_dummy_vector = || {
        let mut vec = Vec::with_capacity(768);
        for i in 0..768 {
            let val = ((i as f64 * 0.013).sin() * 0.1) as f32;
            vec.push(val);
        }
        vec
    };
    
    let embeddings = match req.input {
        serde_json::Value::String(_) => {
            serde_json::json!([make_dummy_vector()])
        }
        serde_json::Value::Array(arr) => {
            let mut list = Vec::new();
            for _ in arr {
                list.push(make_dummy_vector());
            }
            serde_json::json!(list)
        }
        _ => {
            serde_json::json!([make_dummy_vector()])
        }
    };
    
    let response = serde_json::json!({
        "model": state.model_name,
        "embeddings": embeddings
    });
    
    Json(response).into_response()
}

pub async fn handle_create(
    State(_state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: CreateRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let want_stream = req.stream.unwrap_or(true);
    if want_stream {
        let stream = async_stream::stream! {
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "parsing modelfile"}).to_string()));
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "creating model layer"}).to_string()));
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "success"}).to_string()));
        };
        Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        Json(serde_json::json!({
            "status": "success"
        })).into_response()
    }
}

pub async fn handle_copy() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "success"
    }))
}

pub async fn handle_delete() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "success"
    }))
}

pub async fn handle_pull(
    State(_state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: PullRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let want_stream = req.stream.unwrap_or(true);
    if want_stream {
        let stream = async_stream::stream! {
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "pulling manifest"}).to_string()));
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({
                "status": "downloading",
                "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
                "total": 2500000000i64,
                "completed": 2500000000i64
            }).to_string()));
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "success"}).to_string()));
        };
        Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        Json(serde_json::json!({
            "status": "success"
        })).into_response()
    }
}

pub async fn handle_push(
    State(_state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: PushRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let want_stream = req.stream.unwrap_or(true);
    if want_stream {
        let stream = async_stream::stream! {
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "pushing manifest"}).to_string()));
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            yield Ok::<_, anyhow::Error>(format!("{}\n", serde_json::json!({"status": "success"}).to_string()));
        };
        Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        Json(serde_json::json!({
            "status": "success"
        })).into_response()
    }
}

pub async fn handle_show(
    State(state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let _req: ShowRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let response = serde_json::json!({
        "license": "Google Gemma License",
        "modelfile": format!("FROM {}\nSYSTEM \"{}\"", state.model_name, state.system_prompt),
        "parameters": "stop <|im_end|>\nstop <|endoftext|>\n",
        "template": "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n",
        "details": {
            "parent_model": "",
            "format": "tflite",
            "family": "litert",
            "families": ["litert"],
            "parameter_size": "2B",
            "quantization_level": "none"
        }
    });
    
    Json(response).into_response()
}

pub async fn handle_completions(
    State(state): State<AppState>,
    req_body: String,
) -> impl IntoResponse {
    let req: ChatRequest = match serde_json::from_str(&req_body) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };
    
    let want_stream = req.stream.unwrap_or(false);
    let mut sys_msg = state.system_prompt.clone();
    let mut history_arr = serde_json::json!([]);
    let mut current_msg_j = serde_json::json!({
        "role": "user",
        "content": ""
    });
    
    if let Some(messages) = &req.messages {
        if !messages.is_empty() {
            current_msg_j = messages.last().unwrap().clone();
            let len = messages.len();
            for i in 0..len - 1 {
                let msg = &messages[i];
                if msg.get("role").and_then(|r| r.as_str()) == Some("system") {
                    if let Some(c) = msg.get("content").and_then(|c| c.as_str()) {
                        sys_msg = c.to_string();
                    }
                } else {
                    history_arr.as_array_mut().unwrap().push(msg.clone());
                }
            }
        }
    }
    
    let mut opt = serde_json::json!({
        "max_output_tokens": 262144,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40
    });
    if let Some(ref req_opt) = req.options {
        if let Some(t) = req_opt.get("temperature") { opt["temperature"] = t.clone(); }
        if let Some(p) = req_opt.get("top_p") { opt["top_p"] = p.clone(); }
        if let Some(k) = req_opt.get("top_k") { opt["top_k"] = k.clone(); }
        if let Some(m) = req_opt.get("max_output_tokens") { opt["max_output_tokens"] = m.clone(); }
        if let Some(m) = req_opt.get("max_tokens") { opt["max_output_tokens"] = m.clone(); }
        if let Some(m) = req_opt.get("num_predict") { opt["max_output_tokens"] = m.clone(); }
    } else {
        if let Some(t) = req.temperature { opt["temperature"] = serde_json::json!(t); }
        if let Some(p) = req.top_p { opt["top_p"] = serde_json::json!(p); }
        if let Some(k) = req.top_k { opt["top_k"] = serde_json::json!(k); }
        if let Some(m) = req.max_tokens { opt["max_output_tokens"] = serde_json::json!(m); }
        if let Some(m) = req.max_output_tokens { opt["max_output_tokens"] = serde_json::json!(m); }
        if let Some(m) = req.num_predict { opt["max_output_tokens"] = serde_json::json!(m); }
    }
    
    let request_tools = req.tools.clone().map(|t| serde_json::Value::Array(t));
    let history_json_str = history_arr.to_string();
    let current_msg_str = current_msg_j.to_string();
    let config_json_str = opt.to_string();
    let manager = state.manager.clone();
    let model_name = state.model_name.clone();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<ServerStreamEvent>();
    
    tokio::spawn(async move {
        run_agentic_loop(
            manager,
            model_name,
            sys_msg,
            history_json_str,
            current_msg_str,
            Some(config_json_str),
            request_tools,
            event_tx,
        ).await;
    });
    
    if want_stream {
        let model_name = state.model_name.clone();
        let stream = async_stream::stream! {
            let mut last_tool_calls: Option<serde_json::Value> = None;
            while let Some(event) = event_rx.recv().await {
                match event {
                    ServerStreamEvent::Chunk(c) => {
                        let chunk_j = serde_json::json!({
                            "id": "chatcmpl-litert",
                            "object": "chat.completion.chunk",
                            "created": chrono::Utc::now().timestamp(),
                            "model": model_name,
                            "choices": [
                                {
                                    "delta": {
                                        "content": c
                                    },
                                    "finish_reason": serde_json::Value::Null
                                }
                            ]
                        });
                        yield Ok::<_, anyhow::Error>(format!("data: {}\n\n", chunk_j.to_string()));
                    }
                    ServerStreamEvent::ToolCall { raw_tool_calls_json } => {
                        if let Ok(j) = serde_json::from_str::<serde_json::Value>(&raw_tool_calls_json) {
                            if let Some(tc) = j.get("tool_calls") {
                                last_tool_calls = Some(tc.clone());
                            }
                        }
                    }
                    ServerStreamEvent::ToolResult { .. } => {}
                    ServerStreamEvent::Guidance(..) => {}
                    ServerStreamEvent::Error(e) => {
                        yield Err(anyhow::anyhow!(e));
                    }
                    ServerStreamEvent::Done { .. } => {
                        if let Some(tc) = last_tool_calls.take() {
                            let chunk_j = serde_json::json!({
                                "id": "chatcmpl-litert",
                                "object": "chat.completion.chunk",
                                "created": chrono::Utc::now().timestamp(),
                                "model": model_name,
                                "choices": [
                                    {
                                        "delta": {
                                            "tool_calls": tc
                                        },
                                        "finish_reason": "tool_calls"
                                    }
                                ]
                            });
                            yield Ok::<_, anyhow::Error>(format!("data: {}\n\n", chunk_j.to_string()));
                        } else {
                            let chunk_j = serde_json::json!({
                                "id": "chatcmpl-litert",
                                "object": "chat.completion.chunk",
                                "created": chrono::Utc::now().timestamp(),
                                "model": model_name,
                                "choices": [
                                    {
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }
                                ]
                            });
                            yield Ok::<_, anyhow::Error>(format!("data: {}\n\n", chunk_j.to_string()));
                        }
                        yield Ok::<_, anyhow::Error>("data: [DONE]\n\n".to_string());
                    }
                }
            }
        };
        Response::builder()
            .header("Content-Type", "text/event-stream")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let mut final_text = String::new();
        let mut last_tool_calls: Option<serde_json::Value> = None;
        
        while let Some(event) = event_rx.recv().await {
            match event {
                ServerStreamEvent::Chunk(c) => {
                    final_text.push_str(&c);
                }
                ServerStreamEvent::ToolCall { raw_tool_calls_json } => {
                    if let Ok(j) = serde_json::from_str::<serde_json::Value>(&raw_tool_calls_json) {
                        if let Some(tc) = j.get("tool_calls") {
                            last_tool_calls = Some(tc.clone());
                        }
                    }
                }
                ServerStreamEvent::ToolResult { .. } => {}
                ServerStreamEvent::Guidance(..) => {}
                ServerStreamEvent::Error(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
                }
                ServerStreamEvent::Done { .. } => {}
            }
        }
        
        let mut choice = serde_json::json!({
            "message": {
                "role": "assistant",
                "content": final_text
            },
            "finish_reason": "stop"
        });
        if let Some(tc) = last_tool_calls {
            choice = serde_json::json!({
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tc
                },
                "finish_reason": "tool_calls"
            });
        }
        let res_j = serde_json::json!({
            "id": "chatcmpl-litert",
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": state.model_name,
            "choices": vec![choice]
        });
        Json(res_j).into_response()
    }
}
