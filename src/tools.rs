use std::fs;
use std::process::Command;

use crate::utils::{get_workspace_root, log_tool_call, clean_gemma_json};

pub fn execute_dynamic_tool(name: &str, arguments_json: &str, raw_args: &str) -> String {
    let root = get_workspace_root();
    let dynamic_tools_dir = root.join("dynamic_tools");
    let _ = fs::create_dir_all(&dynamic_tools_dir);
    let args_path = dynamic_tools_dir.join(format!("{}_args.json", name));
    if let Err(e) = fs::write(&args_path, arguments_json) {
        let err_msg = format!("Error: Failed to create arguments file for tool execution: {}", e);
        log_tool_call(name, raw_args, arguments_json, -1, &err_msg);
        return err_msg;
    }
    
    let py_path = dynamic_tools_dir.join(format!("{}.py", name));
    let cmd = format!("python3 {} {} 2>&1", py_path.to_string_lossy(), args_path.to_string_lossy());
    let mut command = Command::new("sh");
    command.arg("-c").arg(&cmd);
    
    let output = match command.output() {
        Ok(out) => {
            let exit_code = out.status.code().unwrap_or(-1);
            let stdout_str = String::from_utf8_lossy(&out.stdout).into_owned();
            log_tool_call(name, raw_args, arguments_json, exit_code, &stdout_str);
            stdout_str
        }
        Err(e) => {
            let err_msg = format!("Error: Failed to execute tool command: {}", e);
            log_tool_call(name, raw_args, arguments_json, -1, &err_msg);
            err_msg
        }
    };
    
    let _ = fs::remove_file(&args_path);
    output
}

pub fn execute_tool(name: &str, arguments_json: &str) -> String {
    println!("[시스템] ExecuteTool 호출: name={}, args={}", name, arguments_json);
    let mut args_j: serde_json::Value = match serde_json::from_str(arguments_json) {
        Ok(v) => v,
        Err(e) => {
            let err_msg = format!("Error parsing arguments JSON: {}", e);
            log_tool_call(name, arguments_json, "{}", -1, &err_msg);
            return err_msg;
        }
    };
    clean_gemma_json(&mut args_j);
    let cleaned_args_str = args_j.to_string();
    
    let root = get_workspace_root();
    
    if name == "create_or_update_tool" {
        let tool_name = args_j.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let tool_desc = args_j.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let tool_params = args_j.get("parameters").cloned().unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let raw_code = args_j.get("code").and_then(|v| v.as_str()).unwrap_or("").to_string();
        
        let stripped_code = if raw_code.trim().starts_with("```") {
            let lines: Vec<&str> = raw_code.lines().collect();
            let start = if lines.first().map_or(false, |l| l.trim().starts_with("```")) { 1 } else { 0 };
            let end = if lines.last().map_or(false, |l| l.trim() == "```") { lines.len() - 1 } else { lines.len() };
            lines[start..end].join("\n")
        } else {
            raw_code.clone()
        };
        
        let tool_code = stripped_code
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace("\\\\", "\\");
        
        let tool_code = if !tool_code.contains('\n') && tool_code.matches(';').count() >= 2 {
            tool_code.replace("; ", "\n").replace(";", "\n")
        } else {
            tool_code
        };
        
        println!("[시스템] 서버 측 Tool Call 실행: create_or_update_tool(name: \"{}\")", tool_name);
        println!("[시스템] 도구 코드 미리보기 (처음 300자):\n{}", tool_code.chars().take(300).collect::<String>());
        
        if tool_name.is_empty() || tool_code.is_empty() {
            let err_msg = "{\"status\": \"error\", \"message\": \"도구 이름(name)과 코드(code)는 필수 항목입니다.\"}".to_string();
            log_tool_call(name, arguments_json, &cleaned_args_str, -1, &err_msg);
            return err_msg;
        }
        
        let dynamic_tools_dir = root.join("dynamic_tools");
        let _ = fs::create_dir_all(&dynamic_tools_dir);
        let py_path = dynamic_tools_dir.join(format!("{}.py", tool_name));
        if let Err(e) = fs::write(&py_path, &tool_code) {
            let err_msg = format!("{{\"status\": \"error\", \"message\": \"스크립트 파일 저장 실패: {}\"}}", e);
            log_tool_call(name, arguments_json, &cleaned_args_str, -1, &err_msg);
            return err_msg;
        }
        
        let check_cmd = format!("python3 -m py_compile {} 2>&1", py_path.to_string_lossy());
        let mut check_process = Command::new("sh");
        check_process.arg("-c").arg(&check_cmd);
        
        match check_process.output() {
            Ok(out) => {
                let exit_code = out.status.code().unwrap_or(-1);
                let check_res = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if exit_code != 0 || !check_res.is_empty() {
                    let err_msg = format!("{{\"status\": \"error\", \"message\": \"파이썬 문법 검사 실패: {}\"}}", check_res);
                    log_tool_call(name, arguments_json, &cleaned_args_str, exit_code, &err_msg);
                    return err_msg;
                }
            }
            Err(e) => {
                let err_msg = format!("{{\"status\": \"error\", \"message\": \"컴파일러 구동 실패: {}\"}}", e);
                log_tool_call(name, arguments_json, &cleaned_args_str, -1, &err_msg);
                return err_msg;
            }
        }
        
        let registry_path = dynamic_tools_dir.join("registry.json");
        let mut registry: serde_json::Value = if let Ok(content) = fs::read_to_string(&registry_path) {
            serde_json::from_str(&content).unwrap_or(serde_json::Value::Object(serde_json::Map::new()))
        } else {
            serde_json::Value::Object(serde_json::Map::new())
        };
        
        let mut final_params = tool_params;
        if final_params.is_null() || final_params.as_object().map_or(true, |m| m.is_empty()) {
            final_params = serde_json::json!({
                "type": "object",
                "properties": {},
                "additionalProperties": true
            });
        }

        if let Some(obj) = registry.as_object_mut() {
            obj.insert(tool_name.clone(), serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": final_params
                }
            }));
        }
        
        if let Ok(reg_str) = serde_json::to_string_pretty(&registry) {
            let _ = fs::write(&registry_path, reg_str);
        }
        
        let success_msg = format!("{{\"status\": \"success\", \"message\": \"도구 '{}' 등록 완료. 텍스트를 출력하지 말고 즉시 이 도구를 호출하여 사용자의 원래 요청을 수행하십시오.\"}}", tool_name);
        log_tool_call(name, arguments_json, &cleaned_args_str, 0, &success_msg);
        return success_msg;
    }
    
    let py_path = root.join("dynamic_tools").join(format!("{}.py", name));
    if py_path.exists() {
        println!("[시스템] 서버 측 동적 Tool Call 실행: {}", name);
        return execute_dynamic_tool(name, &cleaned_args_str, arguments_json);
    }
    
    "Unknown tool".to_string()
}
