use litert_lm::LitManager;
use crate::utils::{load_merged_tools, parse_tool_calls};
use std::sync::Arc;
use tokio::sync::mpsc;
use futures_util::StreamExt;

pub enum ServerStreamEvent {
    Chunk(String),
    ToolCall {
        raw_tool_calls_json: String,
    },
    ToolResult {
        name: String,
        result: String,
    },
    Guidance(String),
    Done {
        final_history: Vec<serde_json::Value>,
        prompt_tokens: i32,
        completion_tokens: i32,
    },
    Error(String),
}

fn estimate_tokens(text: &str) -> i32 {
    (text.len() / 4).max(1) as i32
}

fn format_agentic_prompt(
    system_prompt: &str,
    history: &serde_json::Value,
    active_msg: &serde_json::Value,
) -> String {
    let mut prompt = String::new();
    let mut system_added = false;
    
    if let Some(arr) = history.as_array() {
        for msg in arr {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
            
            if role == "system" {
                prompt.push_str(&format!("<start_of_turn>system\n{}<end_of_turn>\n", content));
                system_added = true;
            } else if role == "user" {
                if !system_added && !system_prompt.is_empty() {
                    prompt.push_str(&format!("<start_of_turn>system\n{}<end_of_turn>\n", system_prompt));
                    system_added = true;
                }
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", content));
            } else if role == "assistant" {
                prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", content));
            } else if role == "tool" {
                let name = msg.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                prompt.push_str(&format!("<start_of_turn>user\n[Tool Result for {}]\n{}<end_of_turn>\n", name, content));
            }
        }
    }
    
    if !system_added && !system_prompt.is_empty() {
        prompt.push_str(&format!("<start_of_turn>system\n{}<end_of_turn>\n", system_prompt));
    }
    
    let active_role = active_msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
    let active_content = active_msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
    
    if active_role == "tool" {
        let name = active_msg.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
        prompt.push_str(&format!("<start_of_turn>user\n[Tool Result for {}]\n{}<end_of_turn>\n<start_of_turn>model\n", name, active_content));
    } else {
        prompt.push_str(&format!("<start_of_turn>{}\n{}<end_of_turn>\n<start_of_turn>model\n", 
            if active_role == "assistant" { "model" } else { "user" }, 
            active_content
        ));
    }
    
    prompt
}

pub async fn run_agentic_loop(
    manager: Arc<LitManager>,
    model_name: String,
    system_msg_str: String,
    history_json: String,
    current_msg: String,
    _config_json: Option<String>,
    request_tools: Option<serde_json::Value>,
    event_tx: mpsc::UnboundedSender<ServerStreamEvent>,
) {
    let mut local_history: serde_json::Value = if !history_json.is_empty() {
        serde_json::from_str(&history_json).unwrap_or(serde_json::json!([]))
    } else {
        serde_json::json!([])
    };
    if !local_history.is_array() {
        local_history = serde_json::json!([]);
    }
    
    if let Some(arr) = local_history.as_array_mut() {
        arr.retain(|msg| msg.get("role").and_then(|r| r.as_str()) != Some("system"));
    }
    
    let tools_str = if let Some(ref req_t) = request_tools {
        req_t.to_string()
    } else {
        let loaded = load_merged_tools();
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&loaded) {
            if let Some(arr) = val.as_array() {
                if !arr.is_empty() {
                    println!("  - 로드된 도구 개수: {}개", arr.len());
                }
            }
        }
        loaded
    };
    let tools_opt = if tools_str == "[]" || tools_str.is_empty() { None } else { Some(tools_str) };
    
    let mut system_prompt = system_msg_str.clone();
    if let Some(ref tools) = tools_opt {
        system_prompt.push_str("\n\n사용 가능한 도구 명세 (JSON 포맷):\n");
        system_prompt.push_str(tools);
    }
    
    let active_msg_val = serde_json::from_str::<serde_json::Value>(&current_msg)
        .unwrap_or_else(|_| serde_json::json!({
            "role": "user",
            "content": current_msg.clone()
        }));
        
    let formatted_prompt = format_agentic_prompt(&system_prompt, &local_history, &active_msg_val);
    
    let mut stream = match manager.run_completion_stream(&model_name, &formatted_prompt).await {
        Ok(s) => s,
        Err(e) => {
            let _ = event_tx.send(ServerStreamEvent::Error(format!("{:#}", e)));
            return;
        }
    };
    
    let mut full_response_content = String::new();
    
    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(c) => {
                full_response_content.push_str(&c);
                let _ = event_tx.send(ServerStreamEvent::Chunk(c));
            }
            Err(e) => {
                let _ = event_tx.send(ServerStreamEvent::Error(format!("{:#}", e)));
                return;
            }
        }
    }
    
    let p_tok = estimate_tokens(&formatted_prompt);
    let c_tok = estimate_tokens(&full_response_content);
    
    let tool_calls = parse_tool_calls(&full_response_content);
    if !tool_calls.is_empty() {
        let calls_json = serde_json::json!(tool_calls.iter().map(|tc| {
            serde_json::json!({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments
                }
            })
        }).collect::<Vec<_>>());

        let raw_tc_json = serde_json::json!({
            "tool_calls": calls_json
        }).to_string();
        
        let _ = event_tx.send(ServerStreamEvent::ToolCall {
            raw_tool_calls_json: raw_tc_json,
        });
        
        let mut final_history = local_history.as_array().cloned().unwrap_or_default();
        final_history.push(active_msg_val);
        final_history.push(serde_json::json!({
            "role": "assistant",
            "content": full_response_content,
            "tool_calls": calls_json
        }));
        
        let _ = event_tx.send(ServerStreamEvent::Done {
            final_history,
            prompt_tokens: p_tok,
            completion_tokens: c_tok,
        });
    } else {
        let mut final_history = local_history.as_array().cloned().unwrap_or_default();
        final_history.push(active_msg_val);
        final_history.push(serde_json::json!({
            "role": "assistant",
            "content": full_response_content
        }));
        
        let _ = event_tx.send(ServerStreamEvent::Done {
            final_history,
            prompt_tokens: p_tok,
            completion_tokens: c_tok,
        });
    }
}
