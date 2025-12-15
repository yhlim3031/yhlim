import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify
import easyocr
import firebase_admin
from firebase_admin import credentials, db

# ==== Config ====
HOST = "0.0.0.0"
PORT = 5000
UPLOAD_COOLDOWN = 25  # cooldown dalam saat

# ==== Firebase Init ====
cred = credentials.Certificate(
    r"C:\Users\HP\OneDrive\Documents\smart-attendance\Firebase-admin.json"
)
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://drive-thru-smartattendance-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# ==== Flask App ====
app = Flask(__name__)
reader = easyocr.Reader(['en'])
last_result = {"plate": "-", "time": "-", "method": "none"}
snapshots = []
last_upload_time = {}  # cache cooldown per plate

# ==== Improved OCR ====
def ocr_easyocr(img_bgr):
    try:
        # Improved image preprocessing
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Denoise and enhance contrast
        gray = cv2.medianBlur(gray, 5)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive threshold for better text extraction
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
        
        results = reader.readtext(gray, paragraph=False)
        if not results:
            return "-"
        
        # Filter by confidence and text length
        valid_texts = []
        for res in results:
            text = res[1].strip()
            confidence = res[2]
            
            # Filter: confidence > 0.6 and reasonable length for plates
            if confidence > 0.6 and 3 <= len(text) <= 10:
                # Clean text - keep only alphanumeric
                clean_text = ''.join(c for c in text if c.isalnum()).upper()
                if len(clean_text) >= 3:
                    valid_texts.append(clean_text)
        
        if not valid_texts:
            return "-"
            
        # Return the most likely plate (longest or most plate-like)
        plate_text = max(valid_texts, key=len)
        return plate_text
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return "-"

def detect_and_ocr(img_bgr):
    global last_result, snapshots, last_upload_time
    
    plate = ocr_easyocr(img_bgr)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Skip if no plate detected or invalid plate
    if plate == "-" or len(plate) < 3:
        last_result = {"plate": "-", "time": now_str, "method": "EasyOCR", "status": "No plate detected"}
        return last_result

    # ==== Improved Cooldown Check ====
    current_time = datetime.datetime.now()
    
    # Clean old cooldown entries (older than 1 hour)
    expired_plates = []
    for plate_key, last_time in last_upload_time.items():
        if (current_time - last_time).total_seconds() > 3600:  # 1 hour
            expired_plates.append(plate_key)
    
    for expired_plate in expired_plates:
        del last_upload_time[expired_plate]

    # Check cooldown for current plate
    if plate in last_upload_time:
        elapsed = (current_time - last_upload_time[plate]).total_seconds()
        if elapsed < UPLOAD_COOLDOWN:
            last_result = {
                "plate": plate, 
                "time": now_str, 
                "method": "EasyOCR", 
                "status": f"Cooldown ({int(UPLOAD_COOLDOWN - elapsed)}s remaining)"
            }
            snapshots.insert(0, {"time": now_str, "plate": plate, "img": img_bgr.copy()})
            snapshots = snapshots[:5]
            return last_result

    # Update cooldown timer
    last_upload_time[plate] = current_time

    # ==== Upload to Firebase ====
    last_result = {"plate": plate, "time": now_str, "method": "EasyOCR", "status": "Processing"}
    snapshots.insert(0, {"time": now_str, "plate": plate, "img": img_bgr.copy()})
    snapshots = snapshots[:5]

    save_attendance("plate", plate, now_str)
    print(f"[{now_str}] Plate {plate} processed")
    return last_result

# ==== Improved Attendance Logic ====
def get_user_info_from_plate(plate):
    """Get user info from plate with better error handling"""
    try:
        ref = db.reference(f"plates/{plate}")
        user_data = ref.get()
        
        if not user_data:
            print(f"Plate {plate} not found in database")
            return None
            
        return user_data
    except Exception as e:
        print(f"Error getting user data for plate {plate}: {e}")
        return None

def get_user_info_from_rfid(rfid_uid):
    """Get user info from RFID with proper mapping"""
    try:
        # First get user_id from rfid_to_user mapping
        mapping_ref = db.reference(f"rfid_to_user/{rfid_uid}")
        user_id_from_mapping = mapping_ref.get()
        
        print(f"🔍 [RFID DEBUG] RFID: {rfid_uid} -> User ID dari mapping: {user_id_from_mapping}")
        
        if not user_id_from_mapping:
            print(f"RFID {rfid_uid} not mapped to any user")
            return None
        
        # ⚠️ FIX CASE SENSITIVITY - Cuba berbagai case
        possible_cases = [
            user_id_from_mapping,  # original case
            user_id_from_mapping.lower(),  # semua lowercase
            user_id_from_mapping.upper(),  # semua uppercase
            user_id_from_mapping.capitalize(),  # first letter capital
        ]
        
        # Remove duplicates
        possible_cases = list(set(possible_cases))
        
        print(f"🔍 [RFID DEBUG] Mencari user dengan cases: {possible_cases}")
        
        user_data = None
        actual_user_id = None
        
        for test_case in possible_cases:
            user_ref = db.reference(f"users/{test_case}")
            user_data = user_ref.get()
            if user_data:
                actual_user_id = test_case
                print(f"✅ [RFID DEBUG] User ditemui dengan case: {actual_user_id}")
                break
        
        if not user_data:
            print(f"❌ User tidak ditemui untuk semua case variations: {possible_cases}")
            return None
            
        # ⚠️ Jika case berbeza, update mapping untuk consistency
        if actual_user_id != user_id_from_mapping:
            print(f"⚠️ Case mismatch: {user_id_from_mapping} -> {actual_user_id}")
            # Optional: Update mapping ke case yang betul
            # mapping_ref.set(actual_user_id)
            
        print(f"✅ User data ditemui: {user_data.get('name')}")
        return user_data
        
    except Exception as e:
        print(f"Error getting user data for RFID {rfid_uid}: {e}")
        return None

def determine_shift_and_punctuality(check_time_dt):
    """Determine shift and punctuality based on time"""
    hour = check_time_dt.hour
    minute = check_time_dt.minute
    day_of_week = check_time_dt.weekday()
    
    # Shift determination (Monday-Thursday)
    if day_of_week < 4:
        if hour < 14:
            shift_name = "A"
            shift_start = check_time_dt.replace(hour=8, minute=0, second=0, microsecond=0)
        else:
            shift_name = "B" 
            shift_start = check_time_dt.replace(hour=14, minute=0, second=0, microsecond=0)
    else:  # Friday
        shift_name = "A"
        shift_start = check_time_dt.replace(hour=8, minute=0, second=0, microsecond=0)
    
    # Punctuality check (15 minutes grace period)
    grace_period = shift_start + datetime.timedelta(minutes=15)
    punctuality = "Punctual" if check_time_dt <= grace_period else "Late"
    
    return shift_name, punctuality

def get_minimum_hours(day_of_week):
    """Get minimum required hours based on day"""
    if day_of_week < 4:  # Monday-Thursday
        return 7.0
    elif day_of_week == 4:  # Friday
        return 4.0
    else:  # Weekend
        return 5.0

def save_attendance(mode, key, timestamp):
    today = timestamp.split(" ")[0]
    time_now = timestamp.split(" ")[1]
    time_dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    try:
        # Get user data based on mode
        if mode == "plate":
            user_data = get_user_info_from_plate(key)
            identifier = key
        else:  # rfid mode
            user_data = get_user_info_from_rfid(key)
            identifier = key

        if not user_data:
            print(f"❌ {mode.upper()} {key} tidak didaftarkan. Tiada data disimpan.")
            return

        # ⚠️ PERBAIKI INI: Untuk RFID, GUNA USER ID YANG SUDAH DITEMUI
        if mode == "rfid":
            # User ID sudah ditemui oleh get_user_info_from_rfid, jadi gunakan langsung
            # Dari mapping: jEAl3TFp6NQWUEvt6h3Y9Nxg8wF2
            mapping_ref = db.reference(f"rfid_to_user/{key}")
            user_id = mapping_ref.get()
            
            if not user_id:
                print(f"❌ RFID {key}: User ID tidak ditemui dalam mapping")
                return
                
            print(f"✅ [RFID] Using User ID from mapping: {user_id}")
        else:
            # Untuk plate, kekalkan logic asal
            user_id = user_data.get("user_id") or user_data.get("uid") or f"user_{identifier}"

        name = user_data.get("name", "-")
        jabatan = user_data.get("jabatan", "JABATAN TEKNOLOGI ELEKTRIK DAN ELEKTRONIK")
        
        # Untuk plate mode, gunakan plate yang di-scan
        if mode == "plate":
            plate = key
        else:
            # Untuk RFID, gunakan plate dari user data atau default
            plate = user_data.get("plate", "-")

        # Determine shift and punctuality
        shift_name, punctuality = determine_shift_and_punctuality(time_dt)

        # Check existing attendance - GUNA USER_ID YANG KONSISTEN
        att_ref = db.reference(f"attendance/{today}/{user_id}")
        att_data = att_ref.get()

        print(f"🔍 Checking attendance at: attendance/{today}/{user_id}")
        print(f"🔍 Existing attendance data: {att_data}")

        if not att_data:
            # First check-in
            attendance_record = {
                "user_uid": user_id,
                "name": name,
                "jabatan": jabatan,
                "plate": plate,
                "checkin_plate": plate if mode == "plate" else "-",
                "checkin_method": mode,
                "shift": shift_name,
                "punctuality": punctuality,
                "checkin": time_now,
                "checkout": None,
                "date": today,
                "status": "Checked In",
                "workedHours": "0 hour 0 min",
                "timestamp": timestamp
            }
            att_ref.set(attendance_record)
            print(f"✅ CHECK-IN: {name} at {time_now} | Plate: {plate} | Method: {mode.upper()}")
            
        else:
            # CHECK-OUT PROCESS - Sentiasa update checkout time
            checkin_time_str = att_data.get('checkin', '00:00:00')
            try:
                checkin_time = datetime.datetime.strptime(
                    f"{today} {checkin_time_str}", "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                checkin_time = time_dt - datetime.timedelta(hours=1)  # Fallback
            
            checkout_time = time_dt
            
            # Handle overnight case
            if checkout_time < checkin_time:
                checkout_time += datetime.timedelta(days=1)
            
            # Calculate worked hours
            delta = checkout_time - checkin_time
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            worked_hours_str = f"{hours} hour {minutes} min"
            
            # Determine status based on shift requirements
            min_hours = get_minimum_hours(time_dt.weekday())
            total_hours = delta.total_seconds() / 3600
            status = "Complete" if total_hours >= min_hours else "Incomplete"
            
            # Check if vehicle/method changed
            checkin_plate = att_data.get('checkin_plate', 'Unknown')
            checkin_method = att_data.get('checkin_method', 'unknown')
            change_info = ""
            
            if mode == "plate" and checkin_plate != plate:
                change_info = f" | Vehicle: {checkin_plate} → {plate}"
            elif mode != checkin_method:
                change_info = f" | Method: {checkin_method} → {mode}"
            
            # ⚠️ UBAH INI SAHAJA - Untuk RFID, gunakan RFID ID sebagai checkout_plate
            if mode == "rfid":
                checkout_plate_value = identifier  # "E4F77C05"
            else:
                checkout_plate_value = plate       # "PBL666"
            
            update_data = {
                "checkout": time_now,
                "checkout_method": mode,
                "checkout_plate": checkout_plate_value,  # ⚠️ GUNA VALUE YANG BETUL
                "workedHours": worked_hours_str,
                "status": status,
                "punctuality": att_data.get("punctuality", punctuality),
                "plate": plate,  # Update to current plate
                "timestamp": timestamp
            }
            
            att_ref.update(update_data)
            print(f"✅ CHECK-OUT: {name} at {time_now} | Worked: {worked_hours_str} | Status: {status}{change_info}")

        # Update latest reference
        latest_ref = db.reference("/latestPlate" if mode == "plate" else "/LatestRFID")
        latest_data = {
            "uid": identifier,
            "name": name,
            "date": today,
            "time": time_now,
            "timestamp": timestamp,
            "status": "Success",
            "method": mode.upper()
        }
        
        if mode == "plate":
            latest_data["plate"] = plate
        else:
            latest_data["rfid"] = identifier
            
        latest_ref.set(latest_data)

    except Exception as e:
        print(f"❌ ERROR saving attendance: {e}")
        import traceback
        traceback.print_exc()

# ==== Improved Flask API Endpoints ====
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Smart Attendance API",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if not request.data:
            return jsonify({"error": "No image data provided"}), 400
            
        img_bytes = request.get_data()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Image decode failed"}), 400
        
        result = detect_and_ocr(img)
        return jsonify(result)
        
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/rfid", methods=["POST"])
def rfid():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        uid = data.get("uid", "").strip().upper()
        if not uid:
            return jsonify({"error": "UID required"}), 400
            
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_attendance("rfid", uid, now)
        
        return jsonify({
            "uid": uid, 
            "time": now, 
            "status": "success",
            "message": "RFID attendance recorded successfully"
        })
        
    except Exception as e:
        print(f"RFID error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/status", methods=["GET"])
def status():
    """Check server status and recent activity"""
    return jsonify({
        "status": "running",
        "last_detection": last_result,
        "recent_snapshots": len(snapshots),
        "cooldown_entries": len(last_upload_time),
        "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/debug/clear_cooldown", methods=["POST"])
def clear_cooldown():
    """Debug endpoint to clear cooldown cache"""
    global last_upload_time
    cleared_count = len(last_upload_time)
    last_upload_time = {}
    return jsonify({
        "status": "success", 
        "cleared_entries": cleared_count,
        "message": "Cooldown cache cleared"
    })

# ==== Run Server ====
if __name__ == "__main__":
    print(f"🚀 Starting Smart Attendance Server on {HOST}:{PORT}")
    print(f"📅 Server Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 OCR Language: English")
    print(f"⏰ Upload Cooldown: {UPLOAD_COOLDOWN} seconds")
    app.run(host=HOST, port=PORT, threaded=True, debug=False)