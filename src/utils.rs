use std::fs;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

pub const SOUL_FILE: &str = "soul.txt";
pub const TOOLS_FILE: &str = "tools.txt";

pub const DEFAULT_SOUL: &str = "당신의 이름은 AI입니다.\n한국어만 사용하며, 친절하고 명확하게 답변합니다.\n사용자에게 보이는 답변은 자연스러운 평문을 우선합니다.\n수학적 그래프 시각화가 필요할 경우 수식을 평문으로 설명하고, 필요한 경우 그래프 도구를 사용하세요.";

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String, // Stringified JSON
}

pub fn get_iso8601_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

pub fn get_workspace_root() -> PathBuf {
    let current_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if current_dir.join(SOUL_FILE).exists() || current_dir.join("tools.json").exists() {
        return current_dir;
    }
    let parent_dir = current_dir.join("..");
    if parent_dir.join(SOUL_FILE).exists() || parent_dir.join("tools.json").exists() {
        return parent_dir;
    }
    current_dir
}

pub fn load_system_prompt() -> String {
    let root = get_workspace_root();
    let soul_path = root.join(SOUL_FILE);
    let soul = fs::read_to_string(&soul_path).unwrap_or_else(|_| DEFAULT_SOUL.to_string());
    if soul.trim().is_empty() { DEFAULT_SOUL.to_string() } else { soul }
}

pub fn load_merged_tools() -> String {
    let root = get_workspace_root();
    let tools_json_path = root.join("tools.json");
    let static_tools: serde_json::Value = if let Ok(content) = fs::read_to_string(&tools_json_path) {
        serde_json::from_str(&content).unwrap_or(serde_json::json!([]))
    } else {
        serde_json::json!([])
    };
    let mut static_tools_arr = if let Some(arr) = static_tools.as_array() {
        arr.clone()
    } else {
        vec![]
    };
    
    let registry_path = root.join("dynamic_tools/registry.json");
    let dynamic_registry: serde_json::Value = if let Ok(content) = fs::read_to_string(&registry_path) {
        serde_json::from_str(&content).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };
    
    if let Some(obj) = dynamic_registry.as_object() {
        for (_, spec) in obj {
            static_tools_arr.push(spec.clone());
        }
    }
    
    serde_json::json!(static_tools_arr).to_string()
}

pub fn clean_gemma_json(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::String(s) => {
            if s.starts_with("<|\"|>") {
                *s = s[5..].to_string();
            }
            if s.ends_with("<|\"|>") && s.len() >= 5 {
                *s = s[..s.len() - 5].to_string();
            }
        }
        serde_json::Value::Object(map) => {
            for (_, val) in map.iter_mut() {
                clean_gemma_json(val);
            }
        }
        serde_json::Value::Array(arr) => {
            for val in arr.iter_mut() {
                clean_gemma_json(val);
            }
        }
        _ => {}
    }
}

pub fn extract_text_from_chunk(chunk: &str) -> String {
    if let Ok(j) = serde_json::from_str::<serde_json::Value>(chunk) {
        if let Some(content) = j.get("content") {
            if let Some(s) = content.as_str() {
                return s.to_string();
            }
            if let Some(arr) = content.as_array() {
                let mut res = String::new();
                for item in arr {
                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                        res.push_str(text);
                    }
                }
                return res;
            }
        }
        String::new()
    } else {
        chunk.to_string()
    }
}

pub fn find_all_json_blocks(text: &str) -> Vec<(usize, usize, serde_json::Value)> {
    let mut blocks = Vec::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    
    let mut i = 0;
    while i < len {
        if chars[i] == '{' {
            let mut depth = 1;
            let mut in_string = false;
            let mut escape = false;
            let mut j = i + 1;
            while j < len && depth > 0 {
                let c = chars[j];
                if escape {
                    escape = false;
                } else if c == '\\' {
                    escape = true;
                } else if c == '"' {
                    in_string = !in_string;
                } else if !in_string {
                    if c == '{' {
                        depth += 1;
                    } else if c == '}' {
                        depth -= 1;
                    }
                }
                j += 1;
            }
            
            if depth == 0 {
                let start_byte = text.char_indices().nth(i).map(|(idx, _)| idx).unwrap_or(0);
                let end_byte = text.char_indices().nth(j).map(|(idx, _)| idx).unwrap_or(text.len());
                if let Some(candidate) = text.get(start_byte..end_byte) {
                    let cleaned = candidate
                        .replace("<|\\\"|>", "\\\"")
                        .replace("<|\"|>", "\\\"")
                        .replace("\\\\\"", "\\\"");
                    
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&cleaned) {
                        blocks.push((start_byte, end_byte, parsed));
                        i = j;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    blocks
}

pub fn normalize_tool_calls(val: &serde_json::Value) -> Vec<ToolCall> {
    let mut results = Vec::new();
    if let Some(obj) = val.as_object() {
        // Case 1: Standard OpenAI/Ollama tool_calls list
        if let Some(calls) = obj.get("tool_calls").and_then(|v| v.as_array()) {
            for (idx, call) in calls.iter().enumerate() {
                if let Some(tc) = parse_standard_tool_call(call, idx) {
                    results.push(tc);
                }
            }
            return results;
        }
        
        // Case 2: Standard OpenAI/Ollama single tool call object
        if let Some(tc) = parse_standard_tool_call(val, 0) {
            results.push(tc);
            return results;
        }
        
        // Case 3: Flat tool call or create_or_update_tool
        let has_code = obj.contains_key("code") || obj.contains_key("tool_code");
        let has_name = obj.contains_key("name") || obj.contains_key("tool_name");
        
        if has_name && has_code {
            let name_str = obj.get("name").or(obj.get("tool_name")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let desc_str = obj.get("description").or(obj.get("tool_description")).or(obj.get("desc")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let code_str = obj.get("code").or(obj.get("tool_code")).or(obj.get("script")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            let params_val = obj.get("parameters").or(obj.get("tool_parameters")).cloned().unwrap_or_else(|| {
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": true
                })
            });
            
            let args = serde_json::json!({
                "name": name_str,
                "description": desc_str,
                "code": code_str,
                "parameters": params_val
            });
            
            results.push(ToolCall {
                id: format!("call_create_{}", get_iso8601_now().replace(":", "_").replace("-", "_").replace(".", "_")),
                name: "create_or_update_tool".to_string(),
                arguments: args.to_string(),
            });
            return results;
        }
        
        if has_name {
            let name_str = obj.get("name").or(obj.get("tool_name")).and_then(|v| v.as_str()).unwrap_or("").to_string();
            if !name_str.is_empty() && name_str != "create_or_update_tool" {
                let args_val = if let Some(params) = obj.get("tool_parameters").or(obj.get("parameters")).or(obj.get("arguments")).or(obj.get("args")) {
                    if params.is_object() {
                        params.clone()
                    } else if let Some(s) = params.as_str() {
                        serde_json::from_str(s).unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()))
                    } else {
                        serde_json::Value::Object(serde_json::Map::new())
                    }
                } else {
                    let mut params = serde_json::Map::new();
                    for (k, v) in obj {
                        if k != "name" && k != "tool_name" {
                            params.insert(k.clone(), v.clone());
                        }
                    }
                    serde_json::Value::Object(params)
                };
                
                results.push(ToolCall {
                    id: format!("call_manual_{}", get_iso8601_now().replace(":", "_").replace("-", "_").replace(".", "_")),
                    name: name_str,
                    arguments: args_val.to_string(),
                });
            }
        }
    }
    results
}

fn parse_standard_tool_call(call: &serde_json::Value, index: usize) -> Option<ToolCall> {
    let obj = call.as_object()?;
    let id = obj.get("id").and_then(|v| v.as_str()).unwrap_or_else(|| "call_unknown").to_string();
    let func = obj.get("function")?.as_object()?;
    let name = func.get("name")?.as_str()?.to_string();
    let args_val = func.get("arguments")?;
    let arguments = if let Some(s) = args_val.as_str() {
        s.to_string()
    } else {
        args_val.to_string()
    };
    Some(ToolCall {
        id: if id == "call_unknown" { format!("call_standard_{}_{}", get_iso8601_now().replace(":", "_").replace("-", "_").replace(".", "_"), index) } else { id },
        name,
        arguments,
    })
}

pub fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut results = Vec::new();
    
    // 1. Search for custom tag calls: <|tool_call>call:NAME{ARGS}<tool_call|>
    let mut search_pos = 0;
    while let Some(start_pos) = text[search_pos..].find("<|tool_call>call:") {
        let actual_start = search_pos + start_pos;
        let after_call = &text[actual_start + "<|tool_call>call:".len()..];
        if let Some(brace_pos) = after_call.find('{') {
            let func_name = after_call[..brace_pos].trim().to_string();
            let mut args_str = &after_call[brace_pos..];
            if let Some(end_pos) = args_str.find("<tool_call|>") {
                args_str = &args_str[..end_pos];
            }
            if let Some(last_brace) = args_str.rfind('}') {
                args_str = &args_str[..=last_brace];
            }
            
            let mut cleaned_args = args_str
                .replace("<|\\\"|>", "\"")
                .replace("<|\"|>", "\"")
                .replace("\\\"", "\"")
                .trim()
                .to_string();
                
            while cleaned_args.starts_with("{{") && cleaned_args.ends_with("}}") {
                cleaned_args = cleaned_args[1..cleaned_args.len()-1].trim().to_string();
            }
            
            if let Ok(parsed_json) = serde_json::from_str::<serde_json::Value>(&cleaned_args) {
                results.push(ToolCall {
                    id: format!("call_custom_{}_{}", get_iso8601_now().replace(":", "_").replace("-", "_").replace(".", "_"), results.len()),
                    name: func_name,
                    arguments: parsed_json.to_string(),
                });
            }
        }
        search_pos = actual_start + "<|tool_call>call:".len();
    }
    
    // 2. Parse any JSON blocks for tool calls
    let json_blocks = find_all_json_blocks(text);
    for (_, _, val) in json_blocks {
        let normalized = normalize_tool_calls(&val);
        for tc in normalized {
            if !results.iter().any(|existing| existing.name == tc.name && existing.arguments == tc.arguments) {
                results.push(tc);
            }
        }
    }
    
    results
}

pub fn log_tool_call(name: &str, raw_args: &str, cleaned_args: &str, exit_code: i32, output: &str) {
    let root = get_workspace_root();
    let logs_dir = root.join("logs");
    if let Err(e) = fs::create_dir_all(&logs_dir) {
        eprintln!("[시스템] [오류] 로그 디렉토리 생성 실패: {}", e);
    }
    let log_file_path = logs_dir.join("tool_calls.log");
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_path)
    {
        use std::io::Write;
        let _ = writeln!(file, "========================================");
        let _ = writeln!(file, "시간: {}", get_iso8601_now());
        let _ = writeln!(file, "도구명: {}", name);
        let _ = writeln!(file, "원본 인자: {}", raw_args);
        let _ = writeln!(file, "정제된 인자: {}", cleaned_args);
        let _ = writeln!(file, "종료 코드: {}", exit_code);
        let _ = writeln!(file, "출력/결과:\n{}", output);
        let _ = writeln!(file, "========================================");
    }
}
