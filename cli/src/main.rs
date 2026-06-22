use std::io::{self, Write};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use futures_util::StreamExt;

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANSI escape codes for coloring
    let bold = "\x1b[1m";
    let green = "\x1b[32m";
    let cyan = "\x1b[36m";
    let yellow = "\x1b[33m";
    let magenta = "\x1b[35m";
    let red = "\x1b[31m";
    let reset = "\x1b[0m";

    println!("{}  ___  _ _                        ____ _     ___ ", cyan);
    println!(" / _ \\| | | __ _ _ __ ___   __ _ / ___| |   |_ _|");
    println!("| | | | | |/ _` | '_ ` _ \\ / _` | |   | |    | | ");
    println!("| |_| | | | (_| | | | | | | (_| | |___| |___ | | ");
    println!(" \\___/|_|_|\\__,_|_| |_| |_|\\__,_|\\____|_____|___|{}", reset);
    println!("\n{}LiteRT-LM Ollama CLI 대화형 클라이언트에 오신 것을 환영합니다!{}", bold, reset);
    println!("------------------------------------------------------------");
    println!("* 명령어:");
    println!("  {}/clear{}: 대화 내역 리셋", yellow, reset);
    println!("  {}/exit{}: 클라이언트 종료", yellow, reset);
    println!("------------------------------------------------------------");

    let server_url = "http://localhost:11434/api/chat";
    let model_name = "gemma-4-E2B-it.litertlm";
    let client = Client::new();
    let mut history: Vec<ChatMessage> = Vec::new();

    loop {
        print!("\n{}{}User >{} ", bold, green, reset);
        io::stdout().flush()?;

        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input)?;
        let trimmed = user_input.trim();

        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            "/exit" | "/quit" => {
                println!("{}클라이언트를 종료합니다. 대화해 주셔서 감사합니다!{}", magenta, reset);
                break;
            }
            "/clear" => {
                history.clear();
                println!("{}대화 내역이 초기화되었습니다.{}", yellow, reset);
                continue;
            }
            _ => {}
        }

        // 대화 기록에 사용자 메시지 추가
        history.push(ChatMessage {
            role: "user".to_string(),
            content: trimmed.to_string(),
        });

        let req_body = ChatRequest {
            model: model_name.to_string(),
            messages: history.clone(),
            stream: true,
        };

        print!("\n{}{}Assistant >{} ", bold, cyan, reset);
        io::stdout().flush()?;

        let response = match client.post(server_url)
            .json(&req_body)
            .send()
            .await 
        {
            Ok(resp) => resp,
            Err(e) => {
                println!("\n{}Error: 서버에 연결할 수 없습니다. Ollama 서버가 11434 포트에서 구동 중인지 확인해 주세요. ({}){}", red, e, reset);
                // 사용자 입력 취소
                history.pop();
                continue;
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            println!("\n{}Error: 서버가 오류를 반환했습니다 (Status: {}): {}{}", red, status, text, reset);
            history.pop();
            continue;
        }

        let mut stream = response.bytes_stream();
        let mut assistant_response = String::new();
        let mut buffer = String::new();

        while let Some(item) = stream.next().await {
            match item {
                Ok(bytes) => {
                    let chunk_str = String::from_utf8_lossy(&bytes);
                    buffer.push_str(&chunk_str);

                    // NDJSON 형식이므로 뉴라인으로 구분하여 줄단위로 파싱
                    while let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].trim().to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.is_empty() {
                            continue;
                        }

                        if let Ok(chunk_val) = serde_json::from_str::<Value>(&line) {
                            // guidance 정보 출력
                            if let Some(guidance) = chunk_val.get("guidance").and_then(|g| g.as_str()) {
                                print!("\n{}[시스템 가이드: {}]{} ", yellow, guidance, reset);
                                io::stdout().flush()?;
                            }
                            
                            // tool result 정보 출력
                            if let Some(tr) = chunk_val.get("tool_result") {
                                if let Some(res) = tr.get("result").and_then(|r| r.as_str()) {
                                    let name = tr.get("name").and_then(|n| n.as_str()).unwrap_or("tool");
                                    print!("\n{}[도구 실행 완료 - {}: {}]{} ", yellow, name, res.chars().take(100).collect::<String>(), reset);
                                    io::stdout().flush()?;
                                }
                            }

                            // 일반 텍스트 토큰 출력
                            if let Some(message) = chunk_val.get("message") {
                                if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
                                    print!("{}", content);
                                    io::stdout().flush()?;
                                    assistant_response.push_str(content);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("\n{}Error: 스트림 수신 도중 오류 발생: {}{}", red, e, reset);
                    break;
                }
            }
        }

        println!(); // 마지막 개행 처리

        // 비어 있지 않다면 모델 대화 기록에 추가
        if !assistant_response.is_empty() {
            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: assistant_response,
            });
        }
    }

    Ok(())
}
