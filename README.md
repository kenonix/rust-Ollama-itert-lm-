# litert-lm-server

High-Performance Ollama-Compatible API Server in Rust, powered by GPU-Accelerated Google LiteRT-LM.

`litert-lm-server`는 Google의 LiteRT(TensorFlow Lite) 언어 모델(Gemma 등)을 Rust 환경에서 초고속으로 서빙할 수 있도록 개발된 **Ollama 및 OpenAI 호환 API 서버**입니다. 단일 샷 GPU 백엔드 가속과 표준적인 네이티브 툴 콜링(Tool Calling) 인터페이스를 탑재하고 있습니다.

---

## 🚀 주요 특징 (Key Features)

* **GPU 백엔드 가속 (`--backend gpu`)**:
  - `lit serve` HTTP 서버의 CPU 제한 속도 한계를 극복하기 위해, 내부적으로 GPU 가속 단일 실행 프로세스(`lit run`)를 파이프 스트리밍 방식으로 제어합니다.
  - 이를 통해 쿼리당 반응 속도를 수배 이상 단축시켰습니다.
* **표준 네이티브 툴 콜링 (Standard Native Tool Calling)**:
  - 클라이언트가 API 호출 시 제공한 `tools` 명세 목록을 인식하고, 모델이 도구 사용을 필요로 할 때 표준 Ollama/OpenAI 호환 `tool_calls` 규격으로 응답을 내려줍니다.
  - 불필요한 내부 Python 스크립트 강제 컴파일 및 10회씩 순환하는 자율 에이전트 루프를 제거하여 가볍고 예측 가능한 API 구조를 유지합니다.
* **Ollama 호환성**:
  - `/api/chat` 및 `/api/generate` 엔드포인트를 완벽하게 지원합니다.
  - 스트리밍(`stream: true`) 및 비스트리밍(`stream: false`) 환경 모두 지원합니다.

---

## 🛠 빌드 및 실행 방법 (Build & Run)

### 사전 요구 사항
1. **Rust 툴체인** (Cargo 설치 필수)
2. **GPU 가속 드라이버** (OpenCL / Vulkan 환경)
3. **LiteRT 모델 파일** (예: `gemma-4-E2B-it.litertlm` 모델이 루트 디렉토리에 위치해야 함)

### 빌드
```bash
cargo build --release
```

### 서버 실행
```bash
cargo run --release --bin litert-lm-server -- --port 11434 --model-name gemma-4-E2B-it.litertlm
```

* `--port`: 서버가 수신 대기할 포트 번호 (기본값: `11434`)
* `--model-name`: 사용할 `.litertlm` 모델 파일 이름 (기본값: `gemma-4-E2B-it.litertlm`)

---

## ✉️ API 사용 가이드 (API Usage)

### 1. 일반 대화 스트리밍 테스트
```bash
curl -N http://localhost:11434/api/chat -d '{
  "model": "gemma-4-E2B-it.litertlm",
  "messages": [
    {
      "role": "user",
      "content": "안녕하세요!"
    }
  ],
  "stream": true
}'
```

### 2. 표준 툴 콜링(Tool Calling) 테스트 (비스트리밍)
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "gemma-4-E2B-it.litertlm",
  "messages": [
    {
      "role": "user",
      "content": "plot_function 도구를 사용해서 sin(x) 그래프를 그려줘"
    }
  ],
  "stream": false
}'
```

**응답 예시 (`tool_calls` 반환):**
```json
{
  "model": "gemma-4-E2B-it.litertlm",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "id": "call_manual_2026_06_22T05_48_21_273887Z",
        "type": "function",
        "function": {
          "name": "plot_function",
          "arguments": "{\"expression\":\"sin(x)\",\"title\":\"사인 함수 (sin(x)) 그래프\",\"mode\":\"2d\"}"
        }
      }
    ]
  },
  "done": true
}
```

---

## 📂 프로젝트 구조 (Project Structure)

```
.
├── Cargo.toml
├── README.md               # 본 문서
├── soul.txt                # 시스템 페르소나 설정 파일 (기본 지침)
├── tools.json              # 기본 정적 도구 명세 정의
├── src/
│   ├── main.rs             # 서버 진입점 및 라우팅 설정
│   ├── handlers.rs         # Ollama 호환 핸들러 (Chat, Generate)
│   ├── agentic.rs          # 단일 턴 프롬프트 생성 및 툴 콜링 파서
│   └── utils.rs            # 토큰 추정 및 유틸리티 함수
└── litert-lm/              # LiteRT-LM 모델 바인딩 및 프로세스 풀 제어 모듈
```
