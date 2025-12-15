// oled_rfid_servo_esp32_final_v2.ino
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <ESP32Servo.h>
#include <SPI.h>
#include <MFRC522.h>

// ===== OLED Config =====
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// ===== WiFi Config =====
const char* ssid     = "Redmi 9";
const char* password = "test1234";

// ===== Firebase Config =====
const String FIREBASE_HOST = "drive-thru-smartattendance-default-rtdb.asia-southeast1.firebasedatabase.app";
const String FIREBASE_AUTH = "mb5NvR3JJzhzooXCCoNNQhbxN7ZFLWBEN6OEm7wC";

// ===== Server Flask Config (untuk RFID post) =====
const char* SERVER_URL = "http://192.168.245.172:5000/rfid";

// ===== ESP32-CAM Config =====
const char* ESPCAM_IP = "192.168.245.1"; // IP sebenar ESP32-CAM (kau sahkan)
bool espCamStopSent = false;
bool espCamStartSent = true; // anggap cam mula auto active

// ===== NTP Config =====
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 28800); // GMT+8

// ===== System State =====
enum SystemState { IDLE, DETECTED, GATE_OPEN, GATE_CLOSE };
volatile SystemState state = IDLE;

// ===== Shared Variables =====
String detectedPlate = "";
String plateName = "";
String rfidUID = "";
String rfidName = "";
String displayedName = "";
String detectedDate  = "";
String detectedTime  = "";
String lastDetectionTimestamp = "";
bool plateDetected   = false;
bool rfidDetected    = false;

// ===== debounce / timings =====
unsigned long lastRFIDTime = 0;
const unsigned long RFID_DEBOUNCE_MS = 3000; // ignore same card within 3s
unsigned long lastPlateTime = 0;
const unsigned long PLATE_DEBOUNCE_MS = 3000;

// ===== Scroll OLED =====
int scrollPosition = 0;
unsigned long lastScrollTime = 0;
const unsigned long scrollDelay = 200;

// ===== Servo (MG90S) =====
Servo myServo;
const int servoPin = 13;
bool servoActive = false;
unsigned long servoStartTime = 0;

// ===== Ultrasonic (HC-SR04) =====
const int trigPin = 12;
const int echoPin = 14;
bool ultrasonicActive = false;
const int CLOSE_DISTANCE_CM = 5;
unsigned long lastUltrasonicCheck = 0;
const unsigned long ULTRASONIC_CHECK_INTERVAL = 100; // ms

// ===== RFID (MFRC522) =====
#define RST_PIN 2
#define SS_PIN 5
MFRC522 mfrc(SS_PIN, RST_PIN);

// ===== Firebase poll timing =====
unsigned long lastFirebasePoll = 0;
const unsigned long FIREBASE_POLL_INTERVAL = 400; // ms

// ===== ESP32-CAM command throttle =====
unsigned long lastEspCamCmdAt = 0;
const unsigned long ESPCAM_CMD_MIN_INTERVAL = 400; // ms

// ===== Helpers =====
String uidToString(MFRC522::Uid uid){
  String s = "";
  for(byte i=0;i<uid.size;i++){
    if(uid.uidByte[i]<0x10) s+="0";
    s+=String(uid.uidByte[i], HEX);
  }
  s.toUpperCase();
  return s;
}

long readDistanceCM() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  long duration = pulseIn(echoPin, HIGH, 30000);
  if(duration==0) return -1;
  return duration*0.034/2;
}

int httpGetSimple(const String &url, int timeoutMs = 3000) {
  HTTPClient http;
  http.setTimeout(timeoutMs);
  http.begin(url);
  int code = http.GET();
  http.end();
  return code;
}

// ===== ESP32-CAM Control (build URL on use) =====
void triggerESPCam(bool stop){
  // throttle requests
  if (millis() - lastEspCamCmdAt < ESPCAM_CMD_MIN_INTERVAL) return;
  lastEspCamCmdAt = millis();

  if(WiFi.status() != WL_CONNECTED){
    Serial.println("triggerESPCam(): WiFi not connected");
    return;
  }

  String url = String("http://") + ESPCAM_IP + (stop ? "/stop" : "/start");
  Serial.println("triggerESPCam(): " + url);
  int code = httpGetSimple(url, 3000);
  Serial.println(String("ESPCAM ") + (stop ? "STOP" : "START") + " code: " + String(code));

  if(code >= 200 && code < 300){
    espCamStopSent = stop;
    espCamStartSent = !stop;
  } else {
    Serial.println("triggerESPCam(): request failed (will retry later)");
  }
}

// ===== Gate Control (state-aware) =====
void openGate(){
  if(state != DETECTED){
    Serial.println("openGate(): ignored (state != DETECTED)");
    return;
  }
  Serial.println("openGate(): buka pintu");
  myServo.write(90); // buka
  servoActive = true;
  servoStartTime = millis();
  ultrasonicActive = true;
  triggerESPCam(true); // stop camera while gate open
  state = GATE_OPEN;
}

void closeGate(){
  if(state != GATE_OPEN){
    Serial.println("closeGate(): ignored (state != GATE_OPEN)");
    return;
  }
  Serial.println("closeGate(): tutup pintu");
  myServo.write(0); // tutup
  servoActive = false;
  ultrasonicActive = false;
  triggerESPCam(false); // start camera when gate closed

  // reset detection AFTER camera start request
  plateDetected = false;
  rfidDetected = false;
  detectedPlate = "";
  plateName = "";
  rfidUID = "";
  rfidName = "";
  displayedName = "";
  detectedDate = "";
  detectedTime = "";
  scrollPosition = 0;
  lastDetectionTimestamp = "";

  state = IDLE;
}

// ===== Display handling (tidak diubah visual) =====
void updateOLED(){
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);

  if(servoActive){
    // show plate
    display.setTextSize(2);
    display.setCursor(0,0);
    if(detectedPlate!="") display.println(detectedPlate);

    // show scrolling name
    display.setTextSize(1);
    int16_t x1,y1; uint16_t w,h;
    display.getTextBounds(displayedName,0,0,&x1,&y1,&w,&h);
    int scrollX = -scrollPosition;
    display.setCursor(scrollX,24);
    display.println(displayedName);
    if(millis()-lastScrollTime>scrollDelay){
      lastScrollTime=millis();
      scrollPosition++;
      if(scrollPosition>w+10) scrollPosition=0;
    }

    // date/time
    display.setCursor(0,40);
    display.println(detectedDate);
    display.setCursor(0,52);
    display.println(detectedTime);
  } else {
    // welcome page
    display.setTextSize(2);
    int16_t x1,y1; uint16_t w,h;
    display.getTextBounds("WELCOME",0,0,&x1,&y1,&w,&h);
    display.setCursor((SCREEN_WIDTH-w)/2,0);
    display.println("WELCOME");

    display.setTextSize(1);
    display.getTextBounds("KOLEJ VOKASIONAL",0,0,&x1,&y1,&w,&h);
    display.setCursor((SCREEN_WIDTH-w)/2,25);
    display.println("KOLEJ VOKASIONAL");

    display.getTextBounds("SEBERANG PERAI",0,0,&x1,&y1,&w,&h);
    display.setCursor((SCREEN_WIDTH-w)/2,35);
    display.println("SEBERANG PERAI");

    char timeStr[9];
    time_t epochTime=timeClient.getEpochTime();
    struct tm *ptm=localtime(&epochTime);
    strftime(timeStr,sizeof(timeStr),"%H:%M:%S",ptm);
    display.setTextSize(2);
    display.getTextBounds(timeStr,0,0,&x1,&y1,&w,&h);
    display.setCursor((SCREEN_WIDTH-w)/2,45);
    display.println(timeStr);
  }
  display.display();
}

// ===== Detection handling =====
void handleDetection(const String &nameFromPlate,const String &nameFromRFID,const String &dateStr,const String &timeStr){
  if(nameFromPlate.length()>0 && nameFromRFID.length()>0){
    displayedName=nameFromPlate+" / "+nameFromRFID;
  } else if(nameFromPlate.length()>0){
    displayedName=nameFromPlate;
  } else if(nameFromRFID.length()>0){
    displayedName=nameFromRFID;
  } else displayedName="Unknown";
  detectedDate=dateStr;
  detectedTime=timeStr;
}

// ===== Firebase GET helper =====
String getFirebaseData(String path){
  if(WiFi.status()!=WL_CONNECTED) return "";
  HTTPClient http;
  String url="https://"+FIREBASE_HOST+path+"?auth="+FIREBASE_AUTH;
  http.setTimeout(3000);
  http.begin(url);
  int code=http.GET();
  String payload="";
  if(code==HTTP_CODE_OK) payload=http.getString();
  else Serial.println("getFirebaseData() failed code: " + String(code) + " for " + url);
  http.end();
  return payload;
}

void resetLatestPlate(){
  if(WiFi.status()!=WL_CONNECTED) return;
  HTTPClient http;
  String url="https://"+FIREBASE_HOST+"/latestPlate.json?auth="+FIREBASE_AUTH;
  http.setTimeout(3000);
  http.begin(url);
  int code=http.PUT("null");
  Serial.println("latestPlate reset code:"+String(code));
  http.end();
}

void sendRFIDtoServer(String uid){
  if(WiFi.status()!=WL_CONNECTED) return;
  HTTPClient http;
  http.setTimeout(3000);
  http.begin(SERVER_URL);
  http.addHeader("Content-Type","application/json");
  StaticJsonDocument<128> doc;
  doc["uid"]=uid;
  String body; serializeJson(doc,body);
  int code=http.POST(body);
  Serial.println("RFID send code:"+String(code));
  http.end();
}

// ===== RFID Polling non-blocking =====
void checkRFIDAndProcess(){
  if(state != IDLE) return;

  if(!mfrc.PICC_IsNewCardPresent()) return;
  if(!mfrc.PICC_ReadCardSerial()) return;

  String uid=uidToString(mfrc.uid);

  // debounce same card
  if(millis() - lastRFIDTime < RFID_DEBOUNCE_MS && uid == rfidUID){
    Serial.println("RFID debounced: " + uid);
    mfrc.PICC_HaltA();
    return;
  }
  lastRFIDTime = millis();

  String path="/LatestRFID.json";
  String resp=getFirebaseData(path);
  if(resp!="" && resp!="null"){
    StaticJsonDocument<256> doc;
    DeserializationError err = deserializeJson(doc, resp);
    if(!err){
      String name=String((const char*)doc["name"]);
      time_t epochTime=timeClient.getEpochTime();
      struct tm *ptm=localtime(&epochTime);
      char dateBuf[11], timeBuf[9];
      strftime(dateBuf,sizeof(dateBuf),"%Y-%m-%d",ptm);
      strftime(timeBuf,sizeof(timeBuf),"%H:%M:%S",ptm);

      rfidUID=uid;
      rfidName=name;
      rfidDetected=true;

      // DETECTED -> open gate
      state = DETECTED;
      handleDetection(plateName,rfidName,String(dateBuf),String(timeBuf));
      openGate();
      sendRFIDtoServer(uid);
    } else Serial.println("deserializeJson() error for RFID entry");
  } else Serial.println("RFID not found in Firebase for uid: " + uid);

  mfrc.PICC_HaltA();
}

// ===== Setup =====
void setup(){
  Serial.begin(115200);
  delay(50);
  Serial.println("\n=== ESP32 Gate Controller starting ===");

  // SPI & RFID init
  SPI.begin();
  mfrc.PCD_Init();

  // WiFi
  WiFi.begin(ssid,password);
  Serial.print("Connecting to WiFi");
  unsigned long start = millis();
  while(WiFi.status()!=WL_CONNECTED){
    delay(200);
    Serial.print(".");
    if(millis() - start > 20000){
      Serial.println("\nWiFi connect timeout, restarting...");
      ESP.restart();
    }
  }
  Serial.println("\nWiFi connected: " + WiFi.localIP().toString());

  // OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC,0x3C)){
    Serial.println("SSD1306 allocation failed");
    for(;;);
  }
  display.clearDisplay(); display.display();

  // NTP
  timeClient.begin(); timeClient.setTimeOffset(28800);

  // Servo
  myServo.attach(servoPin); myServo.write(0); // start closed (0°)

  // Ultrasonic pins
  pinMode(trigPin,OUTPUT);
  pinMode(echoPin,INPUT);

  // initial flags/state
  espCamStopSent = false;
  espCamStartSent = true;
  state = IDLE;

  Serial.println("Setup complete");
}

// ===== Main Loop =====
void loop(){
  timeClient.update();

  // 1) RFID
  checkRFIDAndProcess();

  // 2) Firebase plate polling (only when IDLE)
  if(millis()-lastFirebasePoll > FIREBASE_POLL_INTERVAL && state == IDLE){
    lastFirebasePoll = millis();
    String latestPlateJson = getFirebaseData("/latestPlate.json");
    if(latestPlateJson != "" && latestPlateJson != "null"){
      StaticJsonDocument<400> doc;
      DeserializationError err = deserializeJson(doc, latestPlateJson);
      if(!err){
        String plate = String((const char*)doc["plate"]);
        String name  = String((const char*)doc["name"]);
        String dateStr = String((const char*)doc["date"]);
        String timeStr = String((const char*)doc["time"]);
        String timestamp = String((const char*)doc["timestamp"]);

        // debounce repeated plate
        if(timestamp != lastDetectionTimestamp && millis() - lastPlateTime > PLATE_DEBOUNCE_MS){
          lastPlateTime = millis();
          detectedPlate = plate;
          plateName = name;
          detectedDate = dateStr;
          detectedTime = timeStr;
          lastDetectionTimestamp = timestamp;
          plateDetected = true;

          // DETECTED -> open gate
          state = DETECTED;
          handleDetection(plateName, rfidName, detectedDate, detectedTime);
          openGate();
          resetLatestPlate();
        }
      } else Serial.println("deserializeJson() error for latestPlate");
    }
  }

  // 3) Ultrasonic: only while gate open; close immediately when <= 5cm
  if(state == GATE_OPEN && ultrasonicActive && millis()-lastUltrasonicCheck > ULTRASONIC_CHECK_INTERVAL){
    lastUltrasonicCheck = millis();
    long jarak = readDistanceCM();
    if(jarak > 0){
      Serial.println("Ultrasonic distance: " + String(jarak) + " cm");
      if(jarak <= CLOSE_DISTANCE_CM){
        Serial.println("Object detected within close distance -> closing gate");
        closeGate();
      }
    }
  }

  // 4) Update OLED
  updateOLED();

  // small yield
  delay(1);
}
