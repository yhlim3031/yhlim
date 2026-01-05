import datetime
import cv2
import numpy as np
from flask import Flask, request, jsonify
import easyocr
import firebase_admin
from firebase_admin import credentials, db
import os
import uuid
import traceback
from ultralytics import YOLO  # Tambah YOLO

# ==== Config ====
HOST = "0.0.0.0"
PORT = 5000
DUPLICATE_REJECT_WINDOW = 30  # 30 saat reject plat sama
SAVE_DIR = "captured_plates"  # Direktori utama untuk simpan gambar
YOLO_MODEL_PATH = "C:/Users/HP/Downloads/plate.v2i.yolov8/runs/detect/train/weights/best.pt"  # Path ke model YOLO

# ==== Create main save directory if not exists ====
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==== Load YOLO Model ====
print("🔍 Loading YOLO model...")
yolo_model = None
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"✅ YOLO model loaded successfully from: {YOLO_MODEL_PATH}")
    
    # Test model dengan gambar kosong
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = yolo_model(test_img, verbose=False)
    print(f"✅ YOLO model test passed. Model ready for inference.")
    
except Exception as e:
    print(f"⚠️ Failed to load YOLO model: {e}")
    print("ℹ️ System akan menggunakan OCR sahaja (fallback mode)")
    traceback.print_exc()

# ==== Firebase Init ====
try:
    cred = credentials.Certificate(
        r"C:\Users\HP\OneDrive\Documents\smart-attendance\Firebase-admin.json"
    )
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://drive-thru-smartattendance-default-rtdb.asia-southeast1.firebasedatabase.app"
    })
    print("✅ Firebase initialized successfully")
    
    # Test connection immediately
    print("🔍 Testing Firebase connection...")
    test_ref = db.reference("/")
    test_data = test_ref.get()
    if test_data:
        print(f"✅ Firebase connection test PASSED")
        print(f"📊 Root nodes: {list(test_data.keys())}")
        
        # Test access to plates
        plates_ref = db.reference("plates")
        plates_data = plates_ref.get()
        if plates_data:
            print(f"📋 Found {len(plates_data)} plates in database")
            print(f"   Example plates: {list(plates_data.keys())[:5]}")
    else:
        print("⚠️ Firebase connected but no data at root")
        
except Exception as e:
    print(f"❌ Firebase initialization failed: {e}")
    print("🔍 Checking possible issues...")
    print("1. Verify database URL is correct")
    print("2. Check internet connection")
    print("3. Verify Firebase service account file exists")
    traceback.print_exc()

# ==== Flask App ====
app = Flask(__name__)
reader = easyocr.Reader(['en'])
last_result = {"plate": "-", "time": "-", "method": "none"}
snapshots = []

# Sistem reject duplicate
recently_processed = {}  # {plate: {"timestamp": waktu_proses, "count": jumlah_diproses}}

# ==== YOLO Plate Detection Function ====
def detect_plate_yolo(img_bgr):
    """
    Detect plate using YOLO model
    Returns: List of plate regions (crops) and their bounding boxes
    """
    if yolo_model is None:
        print("⚠️ YOLO model not loaded, skipping detection")
        return [], []
    
    try:
        # Run YOLO inference
        results = yolo_model(img_bgr, verbose=False)
        
        plate_crops = []
        plate_boxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Only accept detections with confidence > 0.3
                    if confidence > 0.3:
                        # Ensure coordinates are within image bounds
                        h, w = img_bgr.shape[:2]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        # Crop plate region
                        plate_crop = img_bgr[y1:y2, x1:x2]
                        
                        # Only add if crop is valid
                        if plate_crop.size > 0 and plate_crop.shape[0] > 20 and plate_crop.shape[1] > 50:
                            plate_crops.append(plate_crop)
                            plate_boxes.append((x1, y1, x2, y2, confidence))
                            
                            print(f"✅ Plate detected: Box({x1},{y1},{x2},{y2}) Conf:{confidence:.2f}")
                            print(f"   Crop size: {plate_crop.shape}")
        
        if not plate_crops:
            print("⚠️ No plates detected by YOLO")
            return [], []
        
        print(f"✅ Detected {len(plate_crops)} plate(s)")
        return plate_crops, plate_boxes
        
    except Exception as e:
        print(f"❌ YOLO detection error: {e}")
        traceback.print_exc()
        return [], []

# ==== Original OCR Function (tanpa YOLO) ====
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

# ==== NEW: Hybrid OCR with YOLO + Fallback ====
def ocr_hybrid(img_bgr):
    """
    Try YOLO detection first, if fails use full image OCR
    Returns: plate text and method used
    """
    method = "EasyOCR"
    
    # Try YOLO detection first
    plate_crops, boxes = detect_plate_yolo(img_bgr)
    
    if plate_crops:
        method = "YOLO+EasyOCR"
        print(f"🔍 Using YOLO+OCR method. Found {len(plate_crops)} plate(s)")
        
        all_ocr_results = []
        
        for idx, plate_crop in enumerate(plate_crops):
            print(f"\n🔍 Processing plate crop {idx+1}/{len(plate_crops)}")
            
            # Preprocess crop
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(enhanced, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Run OCR on preprocessed image
            results = reader.readtext(binary, paragraph=False)
            
            if not results:
                print(f"   No text found in plate crop {idx+1}")
                continue
            
            # Process OCR results
            valid_texts = []
            for res in results:
                text = res[1].strip()
                confidence = res[2]
                
                # Filter: confidence > 0.5 and reasonable length for plates
                if confidence > 0.5 and 3 <= len(text) <= 12:
                    # Clean text - keep only alphanumeric
                    clean_text = ''.join(c for c in text if c.isalnum()).upper()
                    if len(clean_text) >= 3:
                        valid_texts.append({
                            "text": clean_text,
                            "confidence": confidence,
                            "bbox": boxes[idx] if idx < len(boxes) else None
                        })
            
            all_ocr_results.extend(valid_texts)
            print(f"   Found {len(valid_texts)} valid text(s) in crop {idx+1}")
        
        if all_ocr_results:
            # Select the best plate text
            all_ocr_results.sort(key=lambda x: (x["confidence"], len(x["text"])), reverse=True)
            
            best_result = all_ocr_results[0]
            plate_text = best_result["text"]
            
            print(f"✅ YOLO+OCR success: '{plate_text}' (Confidence: {best_result['confidence']:.2f})")
            
            # Draw bounding box for debugging
            if best_result["bbox"] is not None:
                x1, y1, x2, y2, conf = best_result["bbox"]
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, f"{plate_text} ({conf:.2f})", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            return plate_text, method
    
    # If YOLO fails or no plates detected, use original OCR
    print("🔍 YOLO failed or no plates detected. Falling back to full image OCR...")
    plate_text = ocr_easyocr(img_bgr)
    method = "EasyOCR (Fallback)"
    
    return plate_text, method

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
        
        # Format components
        date_str = dt.strftime("%Y-%m-%d")      # 2025-12-09
        hour_min = dt.strftime("%H.%M")         # 11.56
        seconds = dt.strftime("%S")             # 54
        
        # Create directory structure: captured_plates/date/plate/
        date_dir = os.path.join(SAVE_DIR, date_str)
        plate_dir = os.path.join(date_dir, clean_plate)
        
        # Create directories if they don't exist
        os.makedirs(plate_dir, exist_ok=True)
        
        # NAMA FAIL: plate_hour.min_seconds.jpg
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
        traceback.print_exc()
        return None

# ==== IMPROVED: Function to get user info from plate ====
def get_user_info_from_plate(plate):
    """Get user info from plate - VERSION DIPERBAIKI"""
    try:
        # Clean the plate number
        clean_plate = ''.join(c for c in plate if c.isalnum()).upper()
        print(f"\n🔍 [PLATE SEARCH] Mencari plat: '{plate}' -> '{clean_plate}'")
        
        # DEBUG: List semua cara plat mungkin disimpan
        print(f"🔍 [PLATE DEBUG] Format kemungkinan untuk plat '{plate}':")
        print(f"   1. Clean: '{clean_plate}' (alphanumeric only)")
        print(f"   2. Original: '{plate}' (as detected)")
        print(f"   3. With spaces variations:")
        print(f"      - '{plate.replace(' ', '')}' (no spaces)")
        print(f"      - '{' '.join(plate.split())}' (single spaces)")
        print(f"      - '{plate.strip()}' (stripped)")
        
        # Get ALL data from Firebase for debugging
        root_ref = db.reference("/")
        all_data = root_ref.get()
        
        if not all_data:
            print(f"❌ [PLATE SEARCH] Firebase kosong!")
            return None
        
        # Debug: Print root structure
        print(f"📊 [PLATE SEARCH] Struktur root: {list(all_data.keys())}")
        
        # OPTION 1: Check in dedicated "plates" directory
        if "plates" in all_data and isinstance(all_data["plates"], dict):
            print(f"📍 [PLATE SEARCH] Mencari di /plates/")
            print(f"   Plat yang ada di /plates/: {list(all_data['plates'].keys())}")
            
            # Check exact match dengan berbagai format
            possible_keys = [
                clean_plate,                    # PBL666
                plate.replace(" ", "").upper(), # PBL666 (no spaces)
                plate.upper().strip(),          # PBL666 (uppercase, stripped)
                ' '.join(plate.upper().split()), # P B L 6 6 6 (normalized spaces)
            ]
            
            # Add variations with different spacing
            if len(plate) >= 3:
                # Try adding spaces: PBL666 -> PBL 666
                if any(c.isdigit() for c in plate) and any(c.isalpha() for c in plate):
                    # Find where numbers start
                    for i in range(1, len(plate)):
                        if plate[i].isdigit() and plate[i-1].isalpha():
                            spaced = plate[:i] + " " + plate[i:]
                            possible_keys.append(spaced.upper().replace(" ", ""))
                            possible_keys.append(spaced.upper())
                            break
            
            # Remove duplicates
            possible_keys = list(set([k for k in possible_keys if k]))
            
            print(f"🔍 [PLATE DEBUG] Mencari dengan keys: {possible_keys}")
            
            for test_key in possible_keys:
                if test_key in all_data["plates"]:
                    user_data = all_data["plates"][test_key]
                    print(f"✅ [PLATE SEARCH] Ditemui di /plates/{test_key}")
                    print(f"   Data: {user_data}")
                    return user_data
            
            # Check partial match (remove spaces, case insensitive)
            for stored_plate, data in all_data["plates"].items():
                if not isinstance(data, dict):
                    continue
                    
                stored_clean = ''.join(c for c in stored_plate if c.isalnum()).upper()
                if clean_plate == stored_clean:
                    print(f"✅ [PLATE SEARCH] Ditemui (partial match): {stored_plate} -> {clean_plate}")
                    print(f"   Data: {data}")
                    return data
            
            print(f"⚠️ [PLATE SEARCH] Plat tidak ditemui di /plates/")
        
        # OPTION 2: Search in "users" directory
        if "users" in all_data and isinstance(all_data["users"], dict):
            print(f"📍 [PLATE SEARCH] Mencari di /users/")
            
            for user_id, user_data in all_data["users"].items():
                if not isinstance(user_data, dict):
                    continue
                    
                # Check various plate field names
                plate_fields = ["plate", "plateNumber", "car_plate", "vehicle_plate", "number_plate", "registration", "car_number"]
                
                for field in plate_fields:
                    if field in user_data:
                        user_plate = user_data.get(field, "")
                        if not user_plate:
                            continue
                            
                        user_plate_str = str(user_plate)
                        user_plate_clean = ''.join(c for c in user_plate_str if c.isalnum()).upper()
                        
                        print(f"   🔍 Checking {user_id}/{field}: '{user_plate_str}' -> '{user_plate_clean}'")
                        
                        if clean_plate == user_plate_clean:
                            print(f"✅ [PLATE SEARCH] Ditemui di /users/{user_id}/{field}")
                            print(f"   Nama: {user_data.get('name', 'Unknown')}")
                            print(f"   Plat asal: '{user_plate_str}' -> Clean: '{user_plate_clean}'")
                            return user_data
        
        # OPTION 3: Deep search in entire database
        print(f"📍 [PLATE SEARCH] Melakukan deep search...")
        
        found_data = None
        found_path = ""
        
        def deep_search(data, current_path=""):
            nonlocal found_data, found_path
            
            if found_data:  # Already found
                return
            
            if isinstance(data, dict):
                # Check for plate fields in this node
                for field in ["plate", "plateNumber", "car_plate", "vehicle_plate", "number_plate", "registration", "carNumber"]:
                    if field in data:
                        field_value = str(data.get(field, ""))
                        if field_value:
                            field_clean = ''.join(c for c in field_value if c.isalnum()).upper()
                            if clean_plate == field_clean:
                                found_data = data
                                found_path = f"{current_path}/{field}"
                                return
                
                # Search recursively in children
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        deep_search(value, f"{current_path}/{key}")
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, (dict, list)):
                        deep_search(item, f"{current_path}[{i}]")
        
        deep_search(all_data, "")
        
        if found_data:
            print(f"✅ [PLATE SEARCH] Ditemui di {found_path}")
            print(f"   Data: {found_data}")
            return found_data
        
        # OPTION 4: Check "rfid_to_user" for plate mappings
        print(f"📍 [PLATE SEARCH] Mencari di rfid_to_user/")
        
        # If we have RFID data, check if any user has this plate
        if "rfid_to_user" in all_data and isinstance(all_data["rfid_to_user"], dict):
            for rfid_uid, user_id in all_data["rfid_to_user"].items():
                if not user_id or not isinstance(user_id, str):
                    continue
                    
                # Get user data
                user_ref = db.reference(f"users/{user_id}")
                user_data = user_ref.get()
                
                if user_data and isinstance(user_data, dict):
                    # Check plate in user data
                    user_plate = user_data.get("plate", "")
                    if user_plate:
                        user_plate_clean = ''.join(c for c in str(user_plate) if c.isalnum()).upper()
                        if clean_plate == user_plate_clean:
                            print(f"✅ [PLATE SEARCH] Ditemui melalui rfid_to_user/{rfid_uid}")
                            print(f"   User: {user_data.get('name', 'Unknown')}")
                            return user_data
        
        print(f"❌ [PLATE SEARCH] Plat '{clean_plate}' tidak ditemui di mana-mana!")
        print(f"📋 [PLATE SEARCH] Cadangan: Pastikan plat didaftarkan di /plates/ atau /users/")
        
        return None
        
    except Exception as e:
        print(f"❌ [PLATE SEARCH] ERROR: {e}")
        traceback.print_exc()
        return None

# ==== NEW: Debug function untuk check plate spacing ====
def debug_plate_spacing(plate):
    """Debug function untuk lihat semua kemungkinan format plat"""
    print(f"\n🔍 [PLATE SPACING DEBUG] Untuk plat: '{plate}'")
    print(f"   Original: '{plate}'")
    print(f"   Upper: '{plate.upper()}'")
    print(f"   Stripped: '{plate.strip()}'")
    print(f"   No spaces: '{plate.replace(' ', '')}'")
    print(f"   Single spaces: '{' '.join(plate.split())}'")
    print(f"   Alphanumeric only: '{''.join(c for c in plate if c.isalnum())}'")
    
    # Generate variations with different spacing patterns
    clean = ''.join(c for c in plate if c.isalnum()).upper()
    if len(clean) >= 3:
        variations = []
        
        # Common Malaysian plate patterns
        if len(clean) <= 7:  # Typical plate length
            # Pattern 1: XXX 1234 (3 letters, space, 4 numbers)
            if len(clean) == 7 and clean[:3].isalpha() and clean[3:].isdigit():
                variations.append(f"{clean[:3]} {clean[3:]}")
            
            # Pattern 2: XX 1234 (2 letters, space, 4 numbers)
            if len(clean) == 6 and clean[:2].isalpha() and clean[2:].isdigit():
                variations.append(f"{clean[:2]} {clean[2:]}")
            
            # Pattern 3: X 1234 (1 letter, space, 4 numbers)
            if len(clean) == 5 and clean[0].isalpha() and clean[1:].isdigit():
                variations.append(f"{clean[0]} {clean[1:]}")
            
            # Pattern 4: 1234 XX (4 numbers, space, 2 letters)
            if len(clean) == 6 and clean[:4].isdigit() and clean[4:].isalpha():
                variations.append(f"{clean[:4]} {clean[4:]}")
        
        print(f"   Common patterns: {variations}")
    
    return clean

# ==== UPDATED: Main detection function with Hybrid approach ====
def detect_and_ocr(img_bgr):
    global last_result, snapshots, recently_processed
    
    # Gunakan OCR hybrid (YOLO + Fallback)
    plate, method = ocr_hybrid(img_bgr)
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Skip if no plate detected or invalid plate
    if plate == "-" or len(plate) < 3:
        last_result = {"plate": "-", "time": now_str, "method": method, "status": "No plate detected"}
        return last_result

    current_time = datetime.datetime.now()
    
    # Debug plate spacing
    debug_plate_spacing(plate)
    
    # PERUBAHAN: REJECT DUPLICATE DALAM 30 SAAT
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
            # DUPLICATE DETECTED WITHIN 30 SECONDS - REJECT!
            reject_count = recently_processed[plate].get("rejected_count", 0) + 1
            recently_processed[plate]["rejected_count"] = reject_count
            
            print(f"❌ GAMBAR DITOLAK: Plat {plate} sudah diproses {elapsed:.1f} saat lepas")
            print(f"   📊 Reject count untuk plat ini: {reject_count}")
            print(f"   ⏰ Tunggu {DUPLICATE_REJECT_WINDOW - elapsed:.1f} saat untuk plat sama")
            
            last_result = {
                "plate": plate, 
                "time": now_str, 
                "method": method, 
                "status": f"REJECTED - Plat sama dalam {DUPLICATE_REJECT_WINDOW}s",
                "image_saved": False,
                "registered": False,
                "reject_reason": f"Duplicate plate detected ({elapsed:.1f}s ago)",
                "reject_count": reject_count,
                "wait_seconds": DUPLICATE_REJECT_WINDOW - elapsed
            }
            
            snapshots.insert(0, {"time": now_str, "plate": f"REJECTED_{plate}", "img": img_bgr.copy()})
            snapshots = snapshots[:5]
            
            # GAMBAR DIBUANG/DELETE - tidak disimpan ke folder
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
        "method": method, 
        "status": "Processing",
        "registered": user_data is not None,
        "image_saved": True if image_path else False,
        "image_path": image_path if image_path else None,
        "filename": os.path.basename(image_path) if image_path else None,
        "folder_structure": f"{SAVE_DIR}/{now_str[:10]}/{''.join(c for c in plate if c.isalnum())}/" if image_path else None,
        "duplicate_protection": {  # Info protection
            "window_seconds": DUPLICATE_REJECT_WINDOW,
            "processed_count": recently_processed[plate].get("processed_count", 1),
            "rejected_count": recently_processed[plate].get("rejected_count", 0),
            "protection_active": True
        }
    }
    
    snapshots.insert(0, {"time": now_str, "plate": plate, "img": img_bgr.copy()})
    snapshots = snapshots[:5]

    # Call save_attendance function
    save_attendance("plate", plate, now_str)
    
    print(f"[{now_str}] Plate {plate} processed - Registered: {user_data is not None}")
    print(f"🛡️ Duplicate protection: Plat ini dilindungi untuk {DUPLICATE_REJECT_WINDOW}s")
    return last_result

# ==== Modified save_attendance - with improved user lookup ====
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
            # Untuk plate, gunakan user_id dari data atau buat berdasarkan plat
            user_id = user_data.get("user_id") or user_data.get("uid") or f"user_{identifier}"
            print(f"✅ [PLATE] User ID: {user_id}")

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
        traceback.print_exc()

# ==== NEW: Function untuk check Firebase connection ====
def check_firebase_connection():
    """Check if Firebase is connected and working"""
    try:
        print("\n" + "="*60)
        print("🔍 FIREBASE CONNECTION CHECK")
        print("="*60)
        
        # Test root access
        root_ref = db.reference("/")
        root_data = root_ref.get()
        
        if root_data is None:
            print("❌ Firebase connected but NO DATA at root")
            print("   Possible issues:")
            print("   1. Database is empty")
            print("   2. Permission denied (check Firebase rules)")
            print("   3. Wrong database URL")
            return False
        
        print(f"✅ Firebase CONNECTED SUCCESSFULLY")
        print(f"📊 Root nodes found: {list(root_data.keys())}")
        
        # Check specific nodes
        nodes_to_check = ["plates", "users", "attendance", "rfid_to_user"]
        for node in nodes_to_check:
            if node in root_data:
                if isinstance(root_data[node], dict):
                    count = len(root_data[node])
                    print(f"   📁 {node}: {count} records")
                else:
                    print(f"   📁 {node}: Present (not a dict)")
            else:
                print(f"   📁 {node}: Not found")
        
        # Test specific plate
        test_plate = "PBL666"
        plates_ref = db.reference(f"plates/{test_plate}")
        plate_data = plates_ref.get()
        
        if plate_data:
            print(f"✅ Test plate '{test_plate}' FOUND")
            print(f"   Name: {plate_data.get('name', 'Unknown')}")
            print(f"   Jabatan: {plate_data.get('jabatan', 'Unknown')}")
        else:
            print(f"⚠️ Test plate '{test_plate}' NOT FOUND")
            # List available plates
            plates_all_ref = db.reference("plates")
            all_plates = plates_all_ref.get()
            if all_plates and isinstance(all_plates, dict):
                print(f"   Available plates: {list(all_plates.keys())}")
        
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ Firebase connection check FAILED: {e}")
        print("Possible issues:")
        print("1. Internet connection")
        print("2. Firebase credentials expired")
        print("3. Database URL incorrect")
        print("4. Firebase project not active")
        traceback.print_exc()
        return False

# ==== NEW DEBUG ENDPOINTS ====

@app.route("/debug/firebase_test", methods=["GET"])
def debug_firebase_test():
    """Debug endpoint untuk test Firebase connection sahaja"""
    try:
        result = check_firebase_connection()
        
        # Get additional info
        root_ref = db.reference("/")
        root_data = root_ref.get()
        
        if result:
            return jsonify({
                "status": "success",
                "firebase_connected": True,
                "message": "Firebase connection successful",
                "root_nodes": list(root_data.keys()) if root_data else [],
                "database_url": "https://drive-thru-smartattendance-default-rtdb.asia-southeast1.firebasedatabase.app",
                "test_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "server_info": {
                    "host": HOST,
                    "port": PORT,
                    "python_version": "3.10"
                }
            })
        else:
            return jsonify({
                "status": "error",
                "firebase_connected": False,
                "message": "Firebase connection failed",
                "database_url": "https://drive-thru-smartattendance-default-rtdb.asia-southeast1.firebasedatabase.app",
                "test_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc(),
            "firebase_connected": False
        }), 500

@app.route("/debug/firebase_structure", methods=["GET"])
def debug_firebase_structure():
    """Debug: Lihat struktur sebenar Firebase"""
    try:
        # Check root structure
        root_ref = db.reference("/")
        all_data = root_ref.get()
        
        if not all_data:
            return jsonify({"error": "Firebase is empty"}), 404
        
        # Find all plates in database
        def find_all_plates(data, path=""):
            plates = []
            if isinstance(data, dict):
                # Check for plate fields
                plate_fields = ["plate", "plateNumber", "car_plate", "vehicle_plate", "number_plate", "registration", "carNumber"]
                for field in plate_fields:
                    if field in data and data[field]:
                        plates.append({
                            "path": f"{path}/{field}",
                            "plate": str(data[field]),
                            "clean_plate": ''.join(c for c in str(data[field]) if c.isalnum()).upper(),
                            "name": data.get("name", "Unknown"),
                            "user_id": data.get("user_id", data.get("uid", "Unknown"))
                        })
                
                # Recursively search
                for key, value in data.items():
                    plates.extend(find_all_plates(value, f"{path}/{key}"))
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    plates.extend(find_all_plates(item, f"{path}[{i}]"))
            
            return plates
        
        all_plates = find_all_plates(all_data, "")
        
        # Get structure
        def get_structure(data, level=0, max_depth=3):
            structure = []
            if isinstance(data, dict):
                for key, value in data.items():
                    if level >= max_depth:
                        structure.append(f"{'  ' * level}{key}: {type(value).__name__}...")
                    else:
                        structure.append(f"{'  ' * level}{key}: {type(value).__name__}")
                        if isinstance(value, (dict, list)):
                            structure.extend(get_structure(value, level + 1, max_depth))
            elif isinstance(data, list):
                structure.append(f"{'  ' * level}List[{len(data)} items]")
                if data and level < max_depth:
                    structure.extend(get_structure(data[0], level + 1, max_depth))
            return structure
        
        structure_lines = get_structure(all_data, max_depth=2)
        
        # Count nodes
        node_count = len(structure_lines)
        
        return jsonify({
            "status": "success",
            "firebase_connected": True,
            "total_nodes": node_count,
            "plates_found": len(all_plates),
            "all_plates": all_plates,
            "root_keys": list(all_data.keys()),
            "firebase_structure": structure_lines[:50],
            "message": f"Found {len(all_plates)} plates in database"
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e), 
            "traceback": traceback.format_exc(),
            "firebase_connected": False
        }), 500

@app.route("/debug/plate_search/<plate>", methods=["GET"])
def debug_plate_search(plate):
    """Debug endpoint untuk test carian plat"""
    try:
        # Debug plate spacing first
        debug_plate_spacing(plate)
        
        result = get_user_info_from_plate(plate)
        
        if result:
            return jsonify({
                "status": "found",
                "plate": plate,
                "clean_plate": ''.join(c for c in plate if c.isalnum()).upper(),
                "data": result,
                "name": result.get("name", "Unknown"),
                "user_id": result.get("user_id", result.get("uid", "Unknown")),
                "message": f"Plate {plate} ditemui dalam database"
            })
        else:
            # Get all plates for reference
            root_ref = db.reference("/")
            all_data = root_ref.get()
            
            available_plates = []
            if "plates" in all_data and isinstance(all_data["plates"], dict):
                available_plates = list(all_data["plates"].keys())
            
            return jsonify({
                "status": "not_found",
                "plate": plate,
                "clean_plate": ''.join(c for c in plate if c.isalnum()).upper(),
                "available_plates": available_plates,
                "available_plates_count": len(available_plates),
                "message": f"Plate {plate} TIDAK ditemui dalam database",
                "suggestion": "Gunakan endpoint /debug/plate_spacing/<plate> untuk debug spacing"
            }), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== NEW: Debug endpoint untuk plate spacing ====
@app.route("/debug/plate_spacing/<plate>", methods=["GET"])
def api_debug_plate_spacing(plate):
    """Debug endpoint untuk lihat semua format kemungkinan plat"""
    try:
        clean_plate = ''.join(c for c in plate if c.isalnum()).upper()
        
        # Generate all possible variations
        variations = {
            "original": plate,
            "uppercase": plate.upper(),
            "lowercase": plate.lower(),
            "stripped": plate.strip(),
            "no_spaces": plate.replace(" ", ""),
            "single_spaces": ' '.join(plate.split()),
            "alphanumeric_only": clean_plate,
        }
        
        # Malaysian plate patterns
        patterns = []
        if len(clean_plate) >= 3:
            # Pattern 1: XXX 1234 (3 letters, space, 4 numbers)
            if len(clean_plate) == 7 and clean_plate[:3].isalpha() and clean_plate[3:].isdigit():
                patterns.append(f"{clean_plate[:3]} {clean_plate[3:]}")
            
            # Pattern 2: XX 1234 (2 letters, space, 4 numbers)
            if len(clean_plate) == 6 and clean_plate[:2].isalpha() and clean_plate[2:].isdigit():
                patterns.append(f"{clean_plate[:2]} {clean_plate[2:]}")
            
            # Pattern 3: X 1234 (1 letter, space, 4 numbers)
            if len(clean_plate) == 5 and clean_plate[0].isalpha() and clean_plate[1:].isdigit():
                patterns.append(f"{clean_plate[0]} {clean_plate[1:]}")
            
            # Pattern 4: 1234 XX (4 numbers, space, 2 letters)
            if len(clean_plate) == 6 and clean_plate[:4].isdigit() and clean_plate[4:].isalpha():
                patterns.append(f"{clean_plate[:4]} {clean_plate[4:]}")
        
        # Check Firebase for each variation
        plates_ref = db.reference("plates")
        all_plates = plates_ref.get() or {}
        
        found_in_firebase = {}
        for key, variation in variations.items():
            if variation in all_plates:
                found_in_firebase[key] = {
                    "variation": variation,
                    "data": all_plates[variation]
                }
        
        # Also check clean version
        if clean_plate in all_plates and "alphanumeric_only" not in found_in_firebase:
            found_in_firebase["alphanumeric_only"] = {
                "variation": clean_plate,
                "data": all_plates[clean_plate]
            }
        
        return jsonify({
            "status": "success",
            "plate": plate,
            "clean_plate": clean_plate,
            "variations": variations,
            "malaysian_patterns": patterns,
            "found_in_firebase": found_in_firebase,
            "all_plates_in_firebase": list(all_plates.keys()),
            "message": f"Generated {len(variations)} variations for plate '{plate}'"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/register_test_plate", methods=["POST"])
def register_test_plate():
    """Register test plate untuk debugging"""
    try:
        data = request.get_json()
        plate = data.get("plate", "TEST123").upper()
        name = data.get("name", "Test User")
        jabatan = data.get("jabatan", "JABATAN TEKNOLOGI ELEKTRIK DAN ELEKTRONIK")
        
        if not plate:
            return jsonify({"error": "Plate required"}), 400
        
        # Clean plate
        clean_plate = ''.join(c for c in plate if c.isalnum())
        
        # Generate user ID
        user_id = f"test_{clean_plate}_{int(datetime.datetime.now().timestamp())}"
        
        # Save to Firebase under plates/
        plates_ref = db.reference(f"plates/{clean_plate}")
        
        user_data = {
            "name": name,
            "plate": plate,
            "clean_plate": clean_plate,
            "jabatan": jabatan,
            "registeredAt": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "uid": user_id,
            "status": "active",
            "department": jabatan,
            "email": f"test_{clean_plate}@example.com"
        }
        
        plates_ref.set(user_data)
        
        # Also save to users/ for consistency
        users_ref = db.reference(f"users/{user_id}")
        users_ref.set(user_data)
        
        return jsonify({
            "status": "success",
            "plate": plate,
            "clean_plate": clean_plate,
            "user_id": user_id,
            "data": user_data,
            "message": f"Test plate {plate} registered successfully",
            "firebase_paths": {
                "plates": f"/plates/{clean_plate}",
                "users": f"/users/{user_id}"
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route("/debug/list_all_plates", methods=["GET"])
def list_all_plates():
    """List semua plat dalam database"""
    try:
        # Check in plates directory
        plates_ref = db.reference("plates")
        plates_data = plates_ref.get()
        
        # Check in users directory
        users_ref = db.reference("users")
        users_data = users_ref.get()
        
        all_plates = {}
        
        # Add from plates/
        if plates_data:
            for plate, data in plates_data.items():
                if isinstance(data, dict):
                    all_plates[plate] = {
                        "source": "plates/",
                        "name": data.get("name", "Unknown"),
                        "user_id": data.get("user_id", data.get("uid", "Unknown")),
                        "jabatan": data.get("jabatan", data.get("department", "Unknown")),
                        "full_data": data
                    }
        
        # Add from users/
        if users_data:
            for user_id, user_data in users_data.items():
                if user_data and isinstance(user_data, dict):
                    plate = user_data.get("plate")
                    if plate:
                        clean_plate = ''.join(c for c in str(plate) if c.isalnum()).upper()
                        all_plates[clean_plate] = {
                            "source": f"users/{user_id}",
                            "name": user_data.get("name", "Unknown"),
                            "user_id": user_id,
                            "jabatan": user_data.get("jabatan", user_data.get("department", "Unknown")),
                            "full_data": user_data
                        }
        
        return jsonify({
            "status": "success",
            "total_plates": len(all_plates),
            "plates": all_plates,
            "message": f"Ditemui {len(all_plates)} plat berdaftar"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/firebase_status", methods=["GET"])
def firebase_status():
    """Check Firebase connection status"""
    try:
        # Test write
        test_ref = db.reference("/_test_connection")
        test_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        test_ref.set({
            "timestamp": test_time,
            "test": "connection_test",
            "server": "smart_attendance"
        })
        
        # Test read
        test_data = test_ref.get()
        
        # Get database stats
        root_ref = db.reference("/")
        root_data = root_ref.get()
        
        return jsonify({
            "status": "connected",
            "firebase_test": "success",
            "test_write_time": test_time,
            "test_read_data": test_data,
            "root_keys": list(root_data.keys()) if isinstance(root_data, dict) else [],
            "database_size": len(str(root_data)) if root_data else 0,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "firebase_connected": False,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }), 500

# ==== Existing endpoints with improvements ====

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
        "yolo_model": "Loaded" if yolo_model else "Not Loaded",
        "ocr_method": "YOLO+OCR (Fallback to OCR only)",
        "recent_snapshots": len(snapshots),
        "protected_plates": protected_plates,
        "protection_window_seconds": DUPLICATE_REJECT_WINDOW,
        "organized_images_saved": total_images,
        "dates_available": date_count,
        "main_directory": os.path.abspath(SAVE_DIR),
        "folder_structure": "captured_plates/YYYY-MM-DD/PLATE_NUMBER/",
        "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "protection_message": f"Reject duplicate plates within {DUPLICATE_REJECT_WINDOW} seconds",
        "processing_flow": "Try YOLO → If fails → Use Full Image OCR",
        "debug_endpoints": {
            "yolo_test": "/test_yolo (POST)",
            "firebase_test": "/debug/firebase_test",
            "firebase_structure": "/debug/firebase_structure",
            "plate_search": "/debug/plate_search/<plate>",
            "plate_spacing": "/debug/plate_spacing/<plate>",
            "list_all_plates": "/debug/list_all_plates",
            "register_test": "/debug/register_test_plate"
        }
    })

# ====== SEMUA FUNGSI ASAL YANG LAIN TETAP SAMA ======

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
        
        # FIX CASE SENSITIVITY - Cuba berbagai case
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
            
        # Jika case berbeza, update mapping untuk consistency
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
        "service": "Smart Attendance API with YOLO Detection",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "yolo_model": "Loaded" if yolo_model else "Not Loaded (Using OCR only)",
        "image_structure": "captured_plates/YYYY-MM-DD/PLATE_NUMBER/image.jpg",
        "duplicate_protection": f"Reject duplicate plates within {DUPLICATE_REJECT_WINDOW} seconds",
        "processing_strategy": "YOLO Detection (if available) → Fallback to Full Image OCR",
        "debug_endpoints": {
            "yolo_test": "/test_yolo (POST image)",
            "firebase_test": "/debug/firebase_test",
            "firebase_structure": "/debug/firebase_structure",
            "plate_search": "/debug/plate_search/<plate>",
            "plate_spacing": "/debug/plate_spacing/<plate>",
            "list_all_plates": "/debug/list_all_plates"
        }
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
        traceback.print_exc()
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
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

# ==== NEW: Test YOLO endpoint ====
@app.route("/test_yolo", methods=["POST"])
def test_yolo_endpoint():
    """Test YOLO detection only"""
    try:
        if not request.data:
            return jsonify({"error": "No image data provided"}), 400
            
        img_bytes = request.get_data()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Image decode failed"}), 400
        
        if yolo_model is None:
            return jsonify({
                "detected": False,
                "message": "YOLO model not loaded",
                "fallback_available": True
            })
        
        # Run YOLO detection
        plate_crops, boxes = detect_plate_yolo(img)
        
        # Draw boxes on image for visualization
        img_with_boxes = img.copy()
        for (x1, y1, x2, y2, conf) in boxes:
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, f"Plate {conf:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save debug image
        debug_dir = os.path.join(SAVE_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"yolo_test_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(debug_path, img_with_boxes)
        
        return jsonify({
            "detected": len(plate_crops) > 0,
            "num_plates": len(plate_crops),
            "boxes": [(int(x1), int(y1), int(x2), int(y2), float(conf)) for (x1, y1, x2, y2, conf) in boxes],
            "crop_sizes": [crop.shape[:2] for crop in plate_crops],
            "image_size": img.shape[:2],
            "model_loaded": True,
            "debug_image": debug_path,
            "message": f"YOLO detected {len(plate_crops)} plate(s)"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==== Run Server ====
if __name__ == "__main__":
    # Run Firebase connection check on startup
    check_firebase_connection()
    
    print(f"""
    🚀 SMART ATTENDANCE SERVER WITH HYBRID DETECTION
    ===============================================
    📅 Server Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    🌐 Host: {HOST}:{PORT}
    🔧 OCR Language: English
    🛡️ Duplicate Protection: {DUPLICATE_REJECT_WINDOW} seconds
    🤖 YOLO Model: {'✅ LOADED' if yolo_model else '❌ NOT LOADED (Using OCR only)'}
    💾 Save Directory: {os.path.abspath(SAVE_DIR)}
    
    📊 HYBRID PROCESSING STRATEGY:
    =================================
    1. 📸 Receive Image
    2. 🤖 Try YOLO Plate Detection
        • Jika YOLO detect plate → Crop region → OCR
        • Jika YOLO gagal → Gunakan Full Image OCR
    3. 🔤 OCR Processing
    4. ✅ Duplicate Protection Check (30s)
    5. ☁️ Firebase Registration Check
    6. 💾 Save Image (Registered Only)
    7. 📝 Update Attendance
    
    ⚠️ IMPORTANT FEATURES:
    =================================
    1. Plat akan DITOLAK jika sama dalam {DUPLICATE_REJECT_WINDOW} saat
    2. Gambar hanya disimpan untuk plat REGISTERED
    3. Fallback ke OCR penuh jika YOLO gagal
    4. Debug endpoints untuk testing
    
    🔍 TEST ENDPOINTS:
    =================================
    POST /test_yolo              - Test YOLO detection sahaja
    GET  /debug/firebase_test    - Test Firebase connection
    GET  /debug/plate_search/ABC - Test plate search
    POST /debug/register_test_plate - Register test plate
    
    🚀 Server starting...
    """)
    
    app.run(host=HOST, port=PORT, threaded=True, debug=False)

    