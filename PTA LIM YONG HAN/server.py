import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify
import easyocr
import firebase_admin
from firebase_admin import credentials, db
import os
import uuid

# ==== Config ====
HOST = "0.0.0.0"
PORT = 5000
DUPLICATE_REJECT_WINDOW = 30  # ⚠️ PERUBAHAN: 30 saat reject plat sama
SAVE_DIR = "captured_plates"  # Direktori utama untuk simpan gambar

# ==== Create main save directory if not exists ====
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==== Firebase Init ====
cred = credentials.Certificate(
    r"C:\Users\HP\OneDrive\Documents\smart-attendance\Firebase-admin.json"
)
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://drive-thru-smattendance-default-rtdb.asia-southeast1.firebasedatabase.app"
})

# ==== Flask App ====
app = Flask(__name__)
reader = easyocr.Reader(['en'])
last_result = {"plate": "-", "time": "-", "method": "none"}
snapshots = []

# ⚠️ PERUBAHAN BESAR: Sistem reject duplicate
recently_processed = {}  # {plate: {"timestamp": waktu_proses, "count": jumlah_diproses}}

# ==== Function to save image ONLY for registered plates ====
def save_registered_plate_image(img_bgr, plate_number, timestamp, user_data):
    """
    Save plate image ONLY if plate is registered and user data exists
    
    Args:
        img_bgr: OpenCV image in BGR format
        plate_number: Detected plate number
        timestamp: Detection timestamp (format: "2024-12-18 14:30:25")
        user_data: User data from Firebase (None if not registered)
    """
    try:
        # ONLY save if plate is registered (user_data exists)
        if not user_data:
            print(f"⚠️ Plat {plate_number} tidak didaftarkan - Gambar TIDAK disimpan")
            return None
        
        # Parse timestamp
        dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Clean plate number
        clean_plate = "".join(c for c in plate_number if c.isalnum())
        if not clean_plate:
            clean_plate = "UNKNOWN"
        
        # ==== STRUKTUR FOLDER BARU ====
        # Format components
        date_str = dt.strftime("%Y-%m-%d")      # 2025-12-09
        hour_min = dt.strftime("%H.%M")         # 11.56
        seconds = dt.strftime("%S")             # 54
        
        # Create directory structure: captured_plates/date/plate/
        date_dir = os.path.join(SAVE_DIR, date_str)
        plate_dir = os.path.join(date_dir, clean_plate)
        
        # Create directories if they don't exist
        os.makedirs(plate_dir, exist_ok=True)
        
        # ==== NAMA FAIL: plate_hour.min_seconds.jpg ====
        filename = f"{clean_plate}_{hour_min}_{seconds}.jpg"
        filepath = os.path.join(plate_dir, filename)
        
        # Check if file already exists (same second)
        counter = 1
        while os.path.exists(filepath):
            # Add counter if file exists
            filename = f"{clean_plate}_{hour_min}_{seconds}_{counter}.jpg"
            filepath = os.path.join(plate_dir, filename)
            counter += 1
        
        # Save image
        success = cv2.imwrite(filepath, img_bgr)
        
        if success:
            # Create thumbnail
            thumbnail = cv2.resize(img_bgr, (320, 240))
            thumb_filename = f"thumb_{filename}"
            thumb_path = os.path.join(plate_dir, thumb_filename)
            cv2.imwrite(thumb_path, thumbnail)
            
            # Get file info
            file_size = os.path.getsize(filepath)
            
            print(f"✅ Gambar plat REGISTERED disimpan")
            print(f"   📁 Lokasi: {plate_dir}")
            print(f"   📄 Fail: {filename}")
            print(f"   📏 Saiz: {file_size} bytes")
            print(f"   👤 Maklumat: {user_data.get('name', 'unknown')} | {plate_number}")
            print(f"   🗂️ Struktur: {SAVE_DIR}/{date_str}/{clean_plate}/{filename}")
            
            return filepath
        else:
            print(f"❌ Gagal menyimpan gambar: {filename}")
            return None
            
    except Exception as e:
        print(f"❌ Error menyimpan gambar: {e}")
        import traceback
        traceback.print_exc()
        return None

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
    global last_result, snapshots, recently_processed
    
    plate = ocr_easyocr(img_bgr)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Skip if no plate detected or invalid plate
    if plate == "-" or len(plate) < 3:
        last_result = {"plate": "-", "time": now_str, "method": "EasyOCR", "status": "No plate detected"}
        return last_result

    current_time = datetime.datetime.now()
    
    # ==== ⚠️ PERUBAHAN BESAR: REJECT DUPLICATE DALAM 30 SAAT ====
    # 1. Clean old entries from recently_processed (> 30 seconds)
    expired_plates = []
    for plate_key, data in recently_processed.items():
        processed_time = data.get("timestamp", current_time)
        if (current_time - processed_time).total_seconds() > DUPLICATE_REJECT_WINDOW:
            expired_plates.append(plate_key)
            print(f"🧹 Cache expired untuk plat {plate_key} (lebih dari {DUPLICATE_REJECT_WINDOW}s)")
    
    for expired_plate in expired_plates:
        if expired_plate in recently_processed:
            del recently_processed[expired_plate]
    
    # 2. Check if this plate was recently processed (within 30 seconds)
    if plate in recently_processed:
        elapsed = (current_time - recently_processed[plate]["timestamp"]).total_seconds()
        
        if elapsed < DUPLICATE_REJECT_WINDOW:
            # ⚠️ DUPLICATE DETECTED WITHIN 30 SECONDS - REJECT!
            reject_count = recently_processed[plate].get("rejected_count", 0) + 1
            recently_processed[plate]["rejected_count"] = reject_count
            
            print(f"❌ GAMBAR DITOLAK: Plat {plate} sudah diproses {elapsed:.1f} saat lepas")
            print(f"   📊 Reject count untuk plat ini: {reject_count}")
            print(f"   ⏰ Tunggu {DUPLICATE_REJECT_WINDOW - elapsed:.1f} saat untuk plat sama")
            
            last_result = {
                "plate": plate, 
                "time": now_str, 
                "method": "EasyOCR", 
                "status": f"REJECTED - Plat sama dalam {DUPLICATE_REJECT_WINDOW}s",
                "image_saved": False,
                "registered": False,
                "reject_reason": f"Duplicate plate detected ({elapsed:.1f}s ago)",
                "reject_count": reject_count,
                "wait_seconds": DUPLICATE_REJECT_WINDOW - elapsed
            }
            
            snapshots.insert(0, {"time": now_str, "plate": f"REJECTED_{plate}", "img": img_bgr.copy()})
            snapshots = snapshots[:5]
            
            # ⚠️ GAMBAR DIBUANG/DELETE - tidak disimpan ke folder
            print(f"🗑️ Gambar untuk plat {plate} DIBUANG (tidak disimpan)")
            return last_result
    
    # 3. Plat belum diproses dalam 30 saat terakhir - PROCESS NORMAL
    # Update recently_processed
    recently_processed[plate] = {
        "timestamp": current_time,
        "processed_count": recently_processed.get(plate, {}).get("processed_count", 0) + 1,
        "rejected_count": 0,
        "last_processed": now_str
    }

    # ==== CHECK IF PLATE IS REGISTERED ====
    user_data = get_user_info_from_plate(plate)
    
    # ==== SAVE IMAGE ONLY IF REGISTERED ====
    image_path = None
    if user_data:
        image_path = save_registered_plate_image(img_bgr.copy(), plate, now_str, user_data)
    
    # ==== Update last_result ====
    last_result = {
        "plate": plate, 
        "time": now_str, 
        "method": "EasyOCR", 
        "status": "Processing",
        "registered": user_data is not None,
        "image_saved": True if image_path else False,
        "image_path": image_path if image_path else None,
        "filename": os.path.basename(image_path) if image_path else None,
        "folder_structure": f"{SAVE_DIR}/{now_str[:10]}/{''.join(c for c in plate if c.isalnum())}/" if image_path else None,
        "duplicate_protection": {  # ⚠️ BARU: Info protection
            "window_seconds": DUPLICATE_REJECT_WINDOW,
            "processed_count": recently_processed[plate].get("processed_count", 1),
            "rejected_count": recently_processed[plate].get("rejected_count", 0),
            "protection_active": True
        }
    }
    
    snapshots.insert(0, {"time": now_str, "plate": plate, "img": img_bgr.copy()})
    snapshots = snapshots[:5]

    # Call save_attendance function (as before)
    save_attendance("plate", plate, now_str)
    
    print(f"[{now_str}] Plate {plate} processed - Registered: {user_data is not None}")
    print(f"🛡️ Duplicate protection: Plat ini dilindungi untuk {DUPLICATE_REJECT_WINDOW}s")
    return last_result

# ==== Modified save_attendance - NO image path in Firebase ====
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

        # Untuk RFID, GUNA USER ID YANG SUDAH DITEMUI
        if mode == "rfid":
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
            
            # Untuk RFID, gunakan RFID ID sebagai checkout_plate
            if mode == "rfid":
                checkout_plate_value = identifier  # "E4F77C05"
            else:
                checkout_plate_value = plate       # "PBL666"
            
            update_data = {
                "checkout": time_now,
                "checkout_method": mode,
                "checkout_plate": checkout_plate_value,
                "workedHours": worked_hours_str,
                "status": status,
                "punctuality": att_data.get("punctuality", punctuality),
                "plate": plate,
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
            latest_data["registered"] = True
        else:
            latest_data["rfid"] = identifier
            
        latest_ref.set(latest_data)

    except Exception as e:
        print(f"❌ ERROR saving attendance: {e}")
        import traceback
        traceback.print_exc()

# ==== New API endpoint untuk duplicate protection info ====
@app.route("/duplicate_protection", methods=["GET"])
def get_duplicate_protection():
    """Get information about duplicate protection"""
    try:
        current_time = datetime.datetime.now()
        protected_plates = []
        
        for plate, data in recently_processed.items():
            elapsed = (current_time - data["timestamp"]).total_seconds()
            if elapsed < DUPLICATE_REJECT_WINDOW:
                protected_plates.append({
                    "plate": plate,
                    "last_processed": data["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "seconds_ago": f"{elapsed:.1f}s",
                    "protected_for": f"{DUPLICATE_REJECT_WINDOW - elapsed:.1f}s",
                    "processed_count": data.get("processed_count", 1),
                    "rejected_count": data.get("rejected_count", 0),
                    "is_protected": elapsed < DUPLICATE_REJECT_WINDOW
                })
        
        # Sort by time (most recent first)
        protected_plates.sort(key=lambda x: float(x["seconds_ago"].replace('s', '')))
        
        return jsonify({
            "status": "success",
            "protection_window_seconds": DUPLICATE_REJECT_WINDOW,
            "total_protected_plates": len(protected_plates),
            "protected_plates": protected_plates,
            "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "protection_message": f"Hanya terima bacaan pertama untuk plat sama dalam {DUPLICATE_REJECT_WINDOW} saat"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/clear_protection_cache", methods=["POST"])
def clear_protection_cache():
    """Debug endpoint to clear protection cache"""
    global recently_processed
    cleared_count = len(recently_processed)
    recently_processed = {}
    
    return jsonify({
        "status": "success", 
        "cleared_plates": cleared_count,
        "message": "Protection cache cleared"
    })

# ==== Organized images endpoint ====
@app.route("/organized_images", methods=["GET"])
def list_organized_images():
    """List all images in organized folder structure"""
    try:
        if not os.path.exists(SAVE_DIR):
            return jsonify({
                "status": "success",
                "total_images": 0,
                "message": "No images directory found"
            })
        
        structure = []
        total_images = 0
        
        # Scan through date directories
        date_dirs = sorted([d for d in os.listdir(SAVE_DIR) 
                          if os.path.isdir(os.path.join(SAVE_DIR, d))], reverse=True)
        
        for date_dir in date_dirs[:10]:  # Last 10 dates only
            date_path = os.path.join(SAVE_DIR, date_dir)
            date_info = {
                "date": date_dir,
                "path": date_path,
                "plates": [],
                "total_images": 0
            }
            
            # Scan through plate directories for this date
            plate_dirs = sorted([p for p in os.listdir(date_path) 
                               if os.path.isdir(os.path.join(date_path, p))])
            
            for plate_dir in plate_dirs:
                plate_path = os.path.join(date_path, plate_dir)
                
                # Count images in this plate directory (excluding thumbnails)
                images = [f for f in os.listdir(plate_path) 
                         if f.endswith('.jpg') and not f.startswith('thumb_')]
                
                if images:
                    plate_info = {
                        "plate": plate_dir,
                        "path": plate_path,
                        "image_count": len(images),
                        "latest_image": max(images, key=lambda x: os.path.getctime(os.path.join(plate_path, x))) if images else None,
                        "images": images[:5]  # First 5 images only
                    }
                    
                    date_info["plates"].append(plate_info)
                    date_info["total_images"] += len(images)
                    total_images += len(images)
            
            if date_info["plates"]:
                structure.append(date_info)
        
        return jsonify({
            "status": "success",
            "total_images": total_images,
            "main_directory": os.path.abspath(SAVE_DIR),
            "folder_structure": "captured_plates/YYYY-MM-DD/PLATE_NUMBER/",
            "example": "captured_plates/2025-12-09/MEDU89/MEDU89_11.56_54.jpg",
            "dates": structure[:5]  # Last 5 dates only
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Endpoint to get image by path ====
@app.route("/images/<date>/<plate>/<filename>", methods=["GET"])
def get_organized_image(date, plate, filename):
    """Get specific image from organized structure"""
    try:
        # Security checks
        if '..' in date or '..' in plate or '..' in filename:
            return jsonify({"error": "Invalid path"}), 400
            
        filepath = os.path.join(SAVE_DIR, date, plate, filename)
        
        if not os.path.exists(filepath):
            return jsonify({"error": "Image not found"}), 404
            
        # Return image as binary
        with open(filepath, 'rb') as f:
            image_data = f.read()
            
        return image_data, 200, {'Content-Type': 'image/jpeg'}
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Updated status endpoint ====
@app.route("/status", methods=["GET"])
def status():
    """Check server status and recent activity"""
    # Count total images in organized structure
    total_images = 0
    date_count = 0
    
    if os.path.exists(SAVE_DIR):
        # Count dates
        dates = [d for d in os.listdir(SAVE_DIR) 
                if os.path.isdir(os.path.join(SAVE_DIR, d))]
        date_count = len(dates)
        
        # Count total images
        for date_dir in dates:
            date_path = os.path.join(SAVE_DIR, date_dir)
            if os.path.isdir(date_path):
                plate_dirs = [p for p in os.listdir(date_path) 
                            if os.path.isdir(os.path.join(date_path, p))]
                for plate_dir in plate_dirs:
                    plate_path = os.path.join(date_path, plate_dir)
                    if os.path.isdir(plate_path):
                        images = [f for f in os.listdir(plate_path) 
                                 if f.endswith('.jpg') and not f.startswith('thumb_')]
                        total_images += len(images)
    
    # Count protected plates (within 30 seconds)
    protected_plates = 0
    current_time = datetime.datetime.now()
    for plate, data in recently_processed.items():
        elapsed = (current_time - data["timestamp"]).total_seconds()
        if elapsed < DUPLICATE_REJECT_WINDOW:
            protected_plates += 1
    
    return jsonify({
        "status": "running",
        "last_detection": last_result,
        "recent_snapshots": len(snapshots),
        "protected_plates": protected_plates,
        "protection_window_seconds": DUPLICATE_REJECT_WINDOW,
        "organized_images_saved": total_images,
        "dates_available": date_count,
        "main_directory": os.path.abspath(SAVE_DIR),
        "folder_structure": "captured_plates/YYYY-MM-DD/PLATE_NUMBER/",
        "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protection_message": f"Reject duplicate plates within {DUPLICATE_REJECT_WINDOW} seconds"
    })

# ====== SEMUA FUNGSI ASAL YANG LAIN TETAP SAMA ======

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

# ==== Improved Flask API Endpoints ====
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Smart Attendance API",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_structure": "captured_plates/YYYY-MM-DD/PLATE_NUMBER/image.jpg",
        "duplicate_protection": f"Reject duplicate plates within {DUPLICATE_REJECT_WINDOW} seconds"
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

# ==== Run Server ====
if __name__ == "__main__":
    print(f"🚀 Starting Smart Attendance Server on {HOST}:{PORT}")
    print(f"📅 Server Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 OCR Language: English")
    print(f"🛡️ Duplicate Protection: {DUPLICATE_REJECT_WINDOW} seconds")
    print(f"   ⚠️ Gambar dengan plat sama dalam {DUPLICATE_REJECT_WINDOW}s akan DIBUANG")
    print(f"💾 Main Directory: {os.path.abspath(SAVE_DIR)}")
    print(f"📸 Struktur folder: {SAVE_DIR}/YYYY-MM-DD/PLATE_NUMBER/")
    print(f"   Contoh: {SAVE_DIR}/2025-12-09/MEDU89/MEDU89_11.56_54.jpg")
    print(f"📸 Hanya simpan gambar untuk plat REGISTERED sahaja")
    app.run(host=HOST, port=PORT, threaded=True, debug=False)