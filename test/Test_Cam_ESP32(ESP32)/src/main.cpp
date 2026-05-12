#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <SPI.h>
#include <MFRC522.h>

// ==========================================================
// 1. CẤU HÌNH MẠNG VÀ SERVER
// ==========================================================
const char* ssid = "92";
const char* password = "1234567891011";

const char* serverName = "http://192.168.1.11:8000/api/swipe"; 

// ==========================================================
// 2. KHAI BÁO CHÂN RFID RC522
// ==========================================================
#define SS_PIN  5
#define RST_PIN 22
MFRC522 rfid(SS_PIN, RST_PIN);

// SCK  D18
// MOSI D23
// MISO D19

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Khởi tạo giao tiếp SPI và module RFID
  SPI.begin();
  rfid.PCD_Init();
  Serial.println("\n[+] Khoi tao RFID RC522 thanh cong. Dang cho the...");

  // Bắt đầu kết nối WiFi
  WiFi.begin(ssid, password);
  Serial.print("[*] Dang ket noi WiFi");
  
  // TỐI ƯU 1: Chống treo mạch lúc mới cấp nguồn (Chờ tối đa 10 giây)
  int retries = 0;
  while (WiFi.status() != WL_CONNECTED && retries < 20) {
    delay(500);
    Serial.print(".");
    retries++;
  }

  // Nếu quá 10 giây không có WiFi -> Tự động khởi động lại mạch (Reset)
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\n[-] Loi WiFi! Dang khoi dong lai ESP32...");
    ESP.restart(); 
  }

  Serial.println("\n[+] Da ket noi WiFi!");
  Serial.print("    IP cua ESP32: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // Kiểm tra xem có thẻ mới đưa vào không
  if (!rfid.PICC_IsNewCardPresent()) return;
  
  // Kiểm tra xem có đọc được dữ liệu thẻ không
  if (!rfid.PICC_ReadCardSerial()) return;

  // Lấy mã UID của thẻ và chuyển thành chuỗi (String)
  String uidString = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) uidString += "0";
    uidString += String(rfid.uid.uidByte[i], HEX);
  }
  uidString.toUpperCase();
  
  Serial.println("\n=====================================");
  Serial.print("[!] Phat hien the UID: ");
  Serial.println(uidString);

  // Halt thẻ hiện tại để module không đọc liên tục cùng 1 thẻ
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  // TỐI ƯU 2: Tự động kết nối lại nếu bị rớt mạng giữa chừng lúc đang chạy
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[-] WiFi bi rot! Dang thu ket noi lai...");
    WiFi.reconnect();
    delay(3000); // Cho mạch 3 giây để dò lại sóng
  }

  // Gửi mã UID lên Python Server qua giao thức HTTP POST
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    // Đóng gói dữ liệu JSON khớp với Backend: {"rfid_code": "XXXXXXXX"}
    String httpRequestData = "{\"rfid_code\":\"" + uidString + "\"}";
    
    Serial.println("[*] Dang gui du lieu len Server...");
    int httpResponseCode = http.POST(httpRequestData);

    // Xử lý phản hồi từ Server Python trả về
    if (httpResponseCode > 0) {
      Serial.print("[+] Server tra ve ma code: ");
      Serial.println(httpResponseCode);
      String payload = http.getString();
      Serial.println("    Noi dung: " + payload);
    } else {
      Serial.print("[-] Loi gui HTTP POST: ");
      Serial.println(httpResponseCode);
    }
    
    // Đóng kết nối HTTP để giải phóng bộ nhớ
    http.end();
  } else {
    // Nếu vẫn không có mạng sau khi đã thử tự cứu hộ
    Serial.println("[-] Loi: He thong dang mat mang, khong the gui the!");
  }
  
  // Trễ 2 giây chống dội thẻ (chống spam quét liên tục)
  delay(2000); 
}