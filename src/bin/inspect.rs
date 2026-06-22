use litert_lm::LitManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("[시스템] LitManager 생성 중...");
    let manager = LitManager::new().await?;
    println!("[시스템] LitManager 생성 성공!");
    
    println!("[시스템] lit 바이너리 확인 및 다운로드 시작...");
    let binary_path = manager.ensure_binary_path().await?;
    println!("[시스템] lit 바이너리 준비 완료: {:?}", binary_path);
    
    Ok(())
}
