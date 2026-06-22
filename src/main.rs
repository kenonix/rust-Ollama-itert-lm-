pub mod utils;
pub mod tools;
pub mod agentic;
pub mod handlers;

use axum::{
    routing::{get, post},
    Router,
};
use axum::http::HeaderValue;
use tower_http::cors::CorsLayer;
use std::sync::Arc;
use anyhow::Result;
use litert_lm::LitManager;

use crate::utils::load_system_prompt;
use crate::handlers::{
    handle_root, handle_version, handle_tags, handle_ps, handle_models,
    handle_chat, handle_generate, handle_embed, handle_create, handle_copy,
    handle_delete, handle_pull, handle_push, handle_show, handle_completions,
};

#[derive(Clone)]
pub struct AppState {
    pub manager: Arc<LitManager>,
    pub system_prompt: String,
    pub model_name: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let mut port = 11434;
    let mut model_name = "gemma-4-E2B-it.litertlm".to_string();
    
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                if i + 1 < args.len() {
                    i += 1;
                    if let Ok(p) = args[i].parse() {
                        port = p;
                    }
                }
            }
            "--model-name" => {
                if i + 1 < args.len() {
                    i += 1;
                    model_name = args[i].clone();
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    println!("[시스템] 시스템 프롬프트 로드 중...");
    let system_prompt = load_system_prompt();
    println!("[시스템] 로드된 시스템 프롬프트:\n{}", system_prompt);
    
    println!("[시스템] LitManager 생성 중...");
    let manager = LitManager::new().await?;
    println!("[시스템] lit 바이너리 확인 및 다운로드 시작...");
    let binary_path = manager.ensure_binary_path().await?;
    println!("[시스템] lit 바이너리 준비 완료: {:?}", binary_path);
    
    let state = AppState {
        manager: Arc::new(manager),
        system_prompt,
        model_name,
    };
    
    let app = Router::new()
        // Ollama API endpoints
        .route("/", get(handle_root))
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/ps", get(handle_ps))
        .route("/api/chat", post(handle_chat))
        .route("/api/generate", post(handle_generate))
        .route("/api/embed", post(handle_embed))
        .route("/api/create", post(handle_create))
        .route("/api/copy", post(handle_copy))
        .route("/api/delete", post(handle_delete))
        .route("/api/pull", post(handle_pull))
        .route("/api/push", post(handle_push))
        .route("/api/show", post(handle_show))
        
        // OpenAI API endpoints
        .route("/v1/models", get(handle_models))
        .route("/models", get(handle_models))
        .route("/v1/chat/completions", post(handle_completions))
        .route("/chat/completions", post(handle_completions))
        
        .layer(
            CorsLayer::new()
                .allow_origin("*".parse::<HeaderValue>().unwrap())
                .allow_methods(tower_http::cors::Any)
                .allow_headers(tower_http::cors::Any),
        )
        .with_state(state);
    
    let addr = format!("0.0.0.0:{}", port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("[서버] {} 주소에서 대기 중...", addr);
    axum::serve(listener, app).await?;
    
    Ok(())
}
