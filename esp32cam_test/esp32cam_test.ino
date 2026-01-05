#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>

// ===== WiFi Config =====
const char* WIFI_SSID     = "Redmi 9";
const char* WIFI_PASSWORD = "test1234";

// ===== Flask server =====
const char* FLASK_UPLOAD_URL = "http://192.168.2.172:5000/upload";

// ===== Camera pins (AI-Thinker) =====
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Pin LED flash
#define FLASH_GPIO_NUM     4

WebServer server(80);
bool cameraActive = false;

void connectWiFi();
void postFrame(void *pvParameters);

void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\nESP32-CAM (NO STREAM) starting...");

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 9;  // BOLEH TURUNKAN (lebih cepat)
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed");
    while (true) delay(1000);
  }

  // Flash OFF pada awal
  pinMode(FLASH_GPIO_NUM, OUTPUT);
  digitalWrite(FLASH_GPIO_NUM, LOW);

  connectWiFi();
  
  // Setup server endpoints (tanpa stream)
  server.on("/", HTTP_GET, []() {
    String html = "<html><head><title>ESP32-CAM</title></head><body>";
    html += "<h2>ESP32-CAM (NO STREAM VERSION)</h2>";
    html += "<p>Camera: <strong>" + String(cameraActive ? "ACTIVE" : "INACTIVE") + "</strong></p>";
    html += "<p>Flash: <strong>" + String(digitalRead(FLASH_GPIO_NUM) ? "ON" : "OFF") + "</strong></p>";
    html += "<p>Posting to Flask every 1 second when active</p>";
    html += "<p><a href='/start'>START Camera</a> | <a href='/stop'>STOP Camera</a></p>";
    html += "</body></html>";
    server.send(200, "text/html", html);
  });
  
  server.on("/start", HTTP_GET, []() {
    digitalWrite(FLASH_GPIO_NUM, HIGH);
    delay(50);
    cameraActive = true;
    server.send(200, "text/plain", "Camera STARTED");
    Serial.println("Camera STARTED - POSTING to Flask");
  });
  
  server.on("/stop", HTTP_GET, []() {
    cameraActive = false;
    delay(20);
    digitalWrite(FLASH_GPIO_NUM, LOW);
    server.send(200, "text/plain", "Camera STOPPED");
    Serial.println("Camera STOPPED");
  });
  
  server.begin();
  
  cameraActive = false;
  
  Serial.println("ESP32-CAM Ready (No Stream)");
  Serial.println("IP: " + WiFi.localIP().toString());

  xTaskCreatePinnedToCore(postFrame, "PostFrameTask", 8192, NULL, 1, NULL, 1);
}

void loop() {
  server.handleClient();
}

void postFrame(void *pvParameters) {
  for (;;) {
    if (cameraActive && WiFi.status() == WL_CONNECTED) {
      camera_fb_t * fb = esp_camera_fb_get();
      if (fb) {
        unsigned long startTime = millis();
        
        HTTPClient http;
        http.begin(FLASK_UPLOAD_URL);
        http.addHeader("Content-Type", "image/jpeg");
        http.setTimeout(1500);  // BOLEH KURANGKAN (1000ms)

        int httpCode = http.POST(fb->buf, fb->len);
        unsigned long endTime = millis();
        
        if (httpCode > 0) {
          Serial.print("POST " + String(fb->len) + " bytes in " + 
                      String(endTime-startTime) + "ms ✓\n");
        } else {
          Serial.print("POST failed in " + String(endTime-startTime) + "ms ✗\n");
        }
        http.end();
        esp_camera_fb_return(fb);
      }
    } else if (!cameraActive) {
      // Camera tidak aktif, idle
      vTaskDelay(100 / portTICK_PERIOD_MS);
    }
    vTaskDelay(1000 / portTICK_PERIOD_MS); // 1 saat antara gambar
  }
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}