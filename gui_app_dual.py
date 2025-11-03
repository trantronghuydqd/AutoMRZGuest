"""
GUI Application - Kéo thả ảnh để đọc MRZ (HỖ TRỢ 2 PASSPORT TRONG 1 ẢNH)
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import os
from passporteye import read_mrz
from datetime import datetime
import re
import cv2
import numpy as np
from PIL import Image

# ============= GUEST MODEL (OOP) =============
class Guest:
    """Object lưu thông tin khách"""
    def __init__(self, full_name, passport_number, dob, gender, issuing_country, nationality, source_image):
        self.full_name = full_name
        self.passport_number = passport_number
        self.dob = dob
        self.gender = gender
        self.issuing_country = issuing_country
        self.nationality = nationality
        self.source_image = source_image
        self.scan_time = datetime.now().strftime("%H:%M:%S")
    
    def __str__(self):
        return f"{self.full_name} - {self.passport_number}"

# ============= IMAGE PREPROCESSING =============

def enhance_mrz_image(image_path):
    """Tăng cường chất lượng ảnh MRZ để OCR tốt hơn"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        height, width = img.shape[:2]
        
        # Crop vùng MRZ (15% dưới cùng)
        mrz_region = img[int(height * 0.85):, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
        
        # Tăng kích thước để OCR tốt hơn
        scale_factor = 3.0
        enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
        
        # Denoise mạnh để giảm nhiễu
        denoised = cv2.fastNlMeansDenoising(enlarged, None, h=10, 
                                           templateWindowSize=7, 
                                           searchWindowSize=21)
        
        # Adaptive threshold để tách chữ rõ hơn
        binary = cv2.adaptiveThreshold(denoised, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Morphology để làm rõ ký tự
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Invert nếu nền đen chữ trắng
        if np.mean(morph) < 127:
            morph = cv2.bitwise_not(morph)
        
        # Lưu ảnh đã enhance
        enhanced_path = image_path.rsplit('.', 1)[0] + '_enhanced.jpg'
        cv2.imwrite(enhanced_path, morph)
        
        return enhanced_path
        
    except Exception as e:
        print(f"Lỗi enhance: {e}")
        return image_path

def rotate_image_if_needed(image_path):
    """Tự động xoay ảnh nếu bị nghiêng hoặc dọc"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        height, width = img.shape[:2]
        
        # Nếu ảnh dọc (chiều cao > chiều rộng), xoay 90 độ
        if height > width:
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotated_path = image_path.rsplit('.', 1)[0] + '_rotated.jpg'
            cv2.imwrite(rotated_path, img_rotated)
            return rotated_path
        
        return image_path
        
    except Exception as e:
        print(f"Lỗi xoay: {e}")
        return image_path

def split_image_horizontal(image_path):
    """Cắt ảnh thành 2 nửa trái/phải (cho 2 passport nằm ngang)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        
        # Cắt làm 2 nửa trái/phải
        mid_x = width // 2
        
        left_half = img[:, :mid_x]
        right_half = img[:, mid_x:]
        
        # Lưu 2 nửa
        base_name = image_path.rsplit('.', 1)[0]
        left_path = f"{base_name}_left.jpg"
        right_path = f"{base_name}_right.jpg"
        
        cv2.imwrite(left_path, left_half)
        cv2.imwrite(right_path, right_half)
        
        return [left_path, right_path]
        
    except Exception as e:
        print(f"Lỗi cắt ảnh: {e}")
        return []

def split_image_vertical(image_path):
    """Cắt ảnh thành 2 nửa trên/dưới (cho 2 passport nằm dọc)"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        
        # Cắt làm 2 nửa trên/dưới
        mid_y = height // 2
        
        top_half = img[:mid_y, :]
        bottom_half = img[mid_y:, :]
        
        # Lưu 2 nửa
        base_name = image_path.rsplit('.', 1)[0]
        top_path = f"{base_name}_top.jpg"
        bottom_path = f"{base_name}_bottom.jpg"
        
        cv2.imwrite(top_path, top_half)
        cv2.imwrite(bottom_path, bottom_half)
        
        return [top_path, bottom_path]
        
    except Exception as e:
        print(f"Lỗi cắt ảnh: {e}")
        return []

def detect_mrz_regions(image_path):
    """Phát hiện các vùng MRZ trong ảnh"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold để tìm vùng text đen
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphology để nhóm text lại
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 10))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Tìm contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Lọc contours có kích thước phù hợp với MRZ
        height, width = img.shape[:2]
        mrz_regions = []
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # MRZ thường ở 15% cuối ảnh, chiều rộng >= 60% ảnh
            if w >= width * 0.5 and y >= height * 0.7:
                mrz_regions.append((x, y, w, h))
        
        # Sắp xếp theo vị trí Y (từ trên xuống)
        mrz_regions.sort(key=lambda r: r[1])
        
        # Nếu tìm thấy >= 2 vùng, có thể là 2 passport
        if len(mrz_regions) >= 2:
            # Crop 2 vùng riêng biệt
            crops = []
            base_name = image_path.rsplit('.', 1)[0]
            
            for i, (x, y, w, h) in enumerate(mrz_regions[:2]):
                # Mở rộng vùng crop 1 chút
                pad_y = int(h * 0.2)
                y_start = max(0, y - pad_y)
                y_end = min(height, y + h + pad_y)
                
                cropped = img[y_start:y_end, x:x+w]
                crop_path = f"{base_name}_passport{i+1}.jpg"
                cv2.imwrite(crop_path, cropped)
                crops.append(crop_path)
            
            return crops
        
        return []
        
    except Exception as e:
        print(f"Lỗi phát hiện MRZ: {e}")
        return []

# ============= MRZ READER =============
def fix_common_ocr_errors(text):
    """Sửa lỗi OCR phổ biến trong MRZ"""
    if not text:
        return ""
    
    # Dictionary các lỗi OCR phổ biến
    corrections = {
        # K vs < confusion
        'I5HIKAWA': 'ISHIKAWA',
        'I5HII': 'ISHII',
        'TAKAHA5HI': 'TAKAHASHI',
        'KAWA5AKI': 'KAWASAKI',
        'KOBA': 'KOBAYASHI',  # Nếu thiếu YASHI
        
        # < được đọc nhầm thành K
        'KKKK': '<<<<',
        'KKK': '<<<',
        'KK': '<<',
        
        # E vs F confusion
        'ENRANCISCO': 'FRANCISCO',
        
        # 0 vs O confusion
        '0SAKA': 'OSAKA',
        'T0KY0': 'TOKYO',
        
        # I vs 1 confusion  
        '1SHIKAWA': 'ISHIKAWA',
        '1SHII': 'ISHII',
        
        # S vs 5 confusion
        'I5HIKAWA': 'ISHIKAWA',
        'I5HII': 'ISHII',
        '5HIMIZU': 'SHIMIZU',
        '5UZUKI': 'SUZUKI',
        '5ATO': 'SATO',
        
        # Common name patterns
        'TAKAHA5HI': 'TAKAHASHI',
        'WATANAKE': 'WATANABE',
        'YAMAM0T0': 'YAMAMOTO',
    }
    
    # Apply corrections
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def clean_name(name):
    """Làm sạch tên với logic thông minh hơn"""
    if not name:
        return ""
    
    # Fix common OCR errors trước
    name = fix_common_ocr_errors(name)
    
    # Xử lý < và K
    # Nếu có nhiều < liên tiếp (2+), giữ nguyên (đây là separator thật)
    # Nếu chỉ có 1 <, có thể là K bị nhầm
    parts = []
    current_part = ""
    
    i = 0
    while i < len(name):
        char = name[i]
        
        # Nếu gặp <<< (3 hoặc nhiều hơn), đây là separator
        if char == '<' and i + 1 < len(name) and name[i + 1] == '<':
            if current_part:
                parts.append(current_part)
                current_part = ""
            # Skip tất cả < liên tiếp
            while i < len(name) and name[i] == '<':
                i += 1
            continue
        
        # Nếu chỉ có 1 < giữa các chữ cái, chuyển thành space
        if char == '<':
            if i > 0 and i < len(name) - 1:
                if name[i-1].isalpha() and name[i+1].isalpha():
                    current_part += ' '
                    i += 1
                    continue
        
        # Nếu có KKK hoặc KK giữa chữ, có thể là <<<
        if char == 'K' and i + 1 < len(name) and name[i + 1] == 'K':
            if current_part:
                parts.append(current_part)
                current_part = ""
            # Skip tất cả K liên tiếp
            while i < len(name) and name[i] == 'K':
                i += 1
            continue
        
        if char.isalpha() or char == ' ':
            current_part += char
        
        i += 1
    
    if current_part:
        parts.append(current_part)
    
    # Join parts và clean up
    name = ' '.join(parts)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Loại bỏ chuỗi ký tự lặp 3 lần trở lên
    name = re.sub(r'(.)\1{2,}', '', name)
    
    return name.strip()

def format_date_from_string(date_str):
    """Chuyển đổi ngày về dd/mm/yyyy"""
    if not date_str:
        return ""
    
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                return date_str
            if len(parts[0]) == 4:
                return f"{parts[2]}/{parts[1]}/{parts[0]}"
    
    if '-' in date_str and len(date_str) == 10:
        parts = date_str.split('-')
        if len(parts[0]) == 4:
            return f"{parts[2]}/{parts[1]}/{parts[0]}"
    
    if len(date_str) == 6 and date_str.isdigit():
        yy = int(date_str[:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        year = 2000 + yy if yy <= 30 else 1900 + yy
        return f"{dd:02d}/{mm:02d}/{year}"
    
    return date_str

def read_single_mrz(image_path, source_name):
    """Đọc MRZ từ 1 ảnh đơn - với enhancement"""
    try:
        # Thử đọc với ảnh gốc
        mrz_obj = read_mrz(image_path)
        
        # Nếu fail, thử enhance
        if not mrz_obj:
            print(f"🔧 Enhance ảnh {source_name}...")
            enhanced = enhance_mrz_image(image_path)
            if enhanced != image_path:
                mrz_obj = read_mrz(enhanced)
                # Xóa file enhanced
                try:
                    os.remove(enhanced)
                except:
                    pass
        
        if not mrz_obj:
            return None
        
        mrz_data = mrz_obj.to_dict()
        if not mrz_data:
            return None
        
        surname = clean_name(mrz_data.get('surname', ''))
        given_names = clean_name(mrz_data.get('names', ''))
        full_name = f"{surname} {given_names}".strip()
        
        sex = mrz_data.get('sex', '')
        gender = 'M' if sex == 'M' else 'F' if sex == 'F' else ''
        
        guest = Guest(
            full_name=full_name,
            passport_number=mrz_data.get('number', ''),
            dob=format_date_from_string(mrz_data.get('date_of_birth', '')),
            gender=gender,
            issuing_country=mrz_data.get('country', ''),
            nationality=mrz_data.get('nationality', ''),
            source_image=source_name
        )
        
        return guest
    except Exception as e:
        print(f"Lỗi đọc MRZ: {e}")
        return None

def read_mrz_from_image(image_path):
    """Đọc MRZ từ ảnh - HỖ TRỢ 2 PASSPORT"""
    guests = []
    temp_files = []
    
    try:
        source_name = os.path.basename(image_path)
        
        # Bước 1: Xoay ảnh nếu cần
        rotated_path = rotate_image_if_needed(image_path)
        if rotated_path != image_path:
            temp_files.append(rotated_path)
        
        # Bước 2: Thử phát hiện nhiều vùng MRZ
        print(f"🔍 Phát hiện vùng MRZ...")
        mrz_crops = detect_mrz_regions(rotated_path)
        
        if mrz_crops:
            print(f"✅ Tìm thấy {len(mrz_crops)} vùng MRZ")
            temp_files.extend(mrz_crops)
            
            for i, crop_path in enumerate(mrz_crops):
                guest = read_single_mrz(crop_path, f"{source_name} (#{i+1})")
                if guest:
                    guests.append(guest)
        
        # Bước 3: Nếu không phát hiện được, thử cắt ngang
        if not guests:
            print(f"🔪 Thử cắt ảnh theo chiều ngang...")
            splits = split_image_horizontal(rotated_path)
            temp_files.extend(splits)
            
            for i, split_path in enumerate(splits):
                guest = read_single_mrz(split_path, f"{source_name} (Left/Right #{i+1})")
                if guest:
                    guests.append(guest)
        
        # Bước 4: Nếu vẫn không được, thử cắt dọc
        if not guests:
            print(f"🔪 Thử cắt ảnh theo chiều dọc...")
            splits = split_image_vertical(rotated_path)
            temp_files.extend(splits)
            
            for i, split_path in enumerate(splits):
                guest = read_single_mrz(split_path, f"{source_name} (Top/Bottom #{i+1})")
                if guest:
                    guests.append(guest)
        
        # Bước 5: Nếu vẫn không được, đọc toàn bộ ảnh
        if not guests:
            print(f"📖 Đọc toàn bộ ảnh...")
            guest = read_single_mrz(rotated_path, source_name)
            if guest:
                guests.append(guest)
        
        return guests
        
    except Exception as e:
        print(f"Lỗi đọc MRZ: {e}")
        return []
    
    finally:
        # Xóa các file tạm
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

# ============= GUI APPLICATION =============
class MRZReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 MRZ Reader - DUAL PASSPORT")
        self.root.geometry("1100x750")
        
        self.guests = []
        self.processing = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Tạo giao diện"""
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=60)
        header.pack(fill=tk.X)
        
        title = tk.Label(header, text="📖 MRZ READER - DUAL PASSPORT SUPPORT", 
                        font=("Arial", 18, "bold"), bg="#2c3e50", fg="white")
        title.pack(pady=10)
        
        # Main container
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Drop Zone & Guest List
        left_frame = tk.Frame(main)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Drop zone
        drop_frame = tk.LabelFrame(left_frame, text="📥 KÉO THẢ ẢNH VÀO ĐÂY", 
                                   font=("Arial", 12, "bold"), bg="#ecf0f1", height=120)
        drop_frame.pack(fill=tk.X, pady=(0, 10))
        drop_frame.pack_propagate(False)
        
        self.drop_label = tk.Label(drop_frame, 
                                   text="🖼️ Kéo thả ảnh (hỗ trợ 2 passport trong 1 ảnh)\n(JPG, PNG, JPEG)",
                                   font=("Arial", 11), bg="#ecf0f1", fg="#7f8c8d")
        self.drop_label.pack(expand=True)
        
        # Enable drag & drop
        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', self.on_drop)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        
        # Guest list
        tk.Label(left_frame, text="📋 DANH SÁCH KHÁCH", 
                font=("Arial", 12, "bold")).pack(pady=5)
        
        columns = ("Name", "Passport", "DOB", "Gender", "Issuing", "Nationality")
        self.tree = ttk.Treeview(left_frame, columns=columns, show="tree headings", height=18)
        
        self.tree.heading("#0", text="STT")
        self.tree.heading("Name", text="Tên")
        self.tree.heading("Passport", text="Passport")
        self.tree.heading("DOB", text="Ngày sinh")
        self.tree.heading("Gender", text="GT")
        self.tree.heading("Issuing", text="Quốc gia cấp")
        self.tree.heading("Nationality", text="Quốc tịch")
        
        self.tree.column("#0", width=40)
        self.tree.column("Name", width=220)
        self.tree.column("Passport", width=120)
        self.tree.column("DOB", width=100)
        self.tree.column("Gender", width=50)
        self.tree.column("Issuing", width=100)
        self.tree.column("Nationality", width=100)
        
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree.bind('<<TreeviewSelect>>', self.on_guest_select)
        self.tree.bind('<Double-1>', self.on_double_click)
        self.tree.bind('<Button-3>', self.show_context_menu)
        self.tree.bind('<Control-c>', self.copy_selected_cell)
        
        # Context menu
        self.context_menu = tk.Menu(self.tree, tearoff=0)
        self.context_menu.add_command(label="📋 Copy ô này", command=self.copy_selected_cell)
        self.context_menu.add_command(label="📋 Copy toàn bộ dòng", command=self.copy_entire_row)
        
        # Right panel
        right_frame = tk.Frame(main, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Status
        status_frame = tk.LabelFrame(right_frame, text="📊 Trạng thái", font=("Arial", 10, "bold"))
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = tk.Label(status_frame, text="⏸️ Sẵn sàng", 
                                     font=("Arial", 12), fg="green")
        self.status_label.pack(pady=10)
        
        self.count_label = tk.Label(status_frame, text="Tổng: 0 khách", 
                                    font=("Arial", 10))
        self.count_label.pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.clear_btn = tk.Button(btn_frame, text="🗑️ XÓA TẤT CẢ", 
                                   command=self.clear_all,
                                   bg="#e74c3c", fg="white", 
                                   font=("Arial", 11, "bold"), height=2)
        self.clear_btn.pack(fill=tk.X, pady=5)
        
        # Selected guest info
        info_frame = tk.LabelFrame(right_frame, text="ℹ️ Thông tin chi tiết", 
                                   font=("Arial", 10, "bold"))
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=10, 
                                                   font=("Courier", 9), 
                                                   state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Action button
        self.fill_btn = tk.Button(right_frame, text="📝 ĐIỀN VÀO SMILE PMS", 
                                  command=self.fill_to_smile,
                                  bg="#3498db", fg="white", 
                                  font=("Arial", 11, "bold"), height=2,
                                  state=tk.DISABLED)
        self.fill_btn.pack(fill=tk.X, pady=10)
        
        # Log
        log_frame = tk.LabelFrame(right_frame, text="📝 Log", font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, 
                                                  font=("Courier", 8))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log("✅ Sẵn sàng nhận ảnh")
        self.log("💡 Hỗ trợ 2 passport trong 1 ảnh!")
    
    def on_drop(self, event):
        """Xử lý khi kéo thả file"""
        if self.processing:
            self.log("⚠️ Đang xử lý, vui lòng đợi...")
            return
        
        # Parse file paths
        files = self.root.tk.splitlist(event.data)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            self.log("❌ Không có file ảnh hợp lệ")
            return
        
        self.log(f"📥 Nhận {len(image_files)} ảnh")
        
        # Process in thread
        threading.Thread(target=self.process_images, args=(image_files,), daemon=True).start()
    
    def process_images(self, image_files):
        """Xử lý nhiều ảnh"""
        self.processing = True
        self.status_label.config(text="⏳ Đang xử lý...", fg="orange")
        
        for image_path in image_files:
            try:
                self.log(f"📸 Đọc: {os.path.basename(image_path)}")
                
                guests = read_mrz_from_image(image_path)
                
                if guests:
                    for guest in guests:
                        self.add_guest(guest)
                        self.log(f"✅ {guest.full_name} - {guest.passport_number}")
                    
                    if len(guests) > 1:
                        self.log(f"🎉 Tìm thấy {len(guests)} passport trong ảnh!")
                else:
                    self.log(f"❌ Không đọc được MRZ: {os.path.basename(image_path)}")
            
            except Exception as e:
                self.log(f"❌ Lỗi: {e}")
        
        self.processing = False
        self.status_label.config(text="✅ Hoàn thành", fg="green")
        self.log("🎉 Xử lý xong!")
    
    def add_guest(self, guest):
        """Thêm guest vào list"""
        self.guests.append(guest)
        
        index = len(self.guests)
        self.tree.insert("", tk.END, text=str(index),
                        values=(guest.full_name, 
                               guest.passport_number,
                               guest.dob,
                               guest.gender,
                               guest.issuing_country,
                               guest.nationality))
        
        self.count_label.config(text=f"Tổng: {len(self.guests)} khách")
    
    def on_guest_select(self, event):
        """Khi chọn guest"""
        selection = self.tree.selection()
        if not selection:
            self.fill_btn.config(state=tk.DISABLED)
            return
        
        item = self.tree.item(selection[0])
        index = int(item['text']) - 1
        
        if 0 <= index < len(self.guests):
            guest = self.guests[index]
            
            info = f"""
╔══════════════════════════════════╗
  THÔNG TIN KHÁCH #{index + 1}
╚══════════════════════════════════╝

👤 Tên: {guest.full_name}
🛂 Passport: {guest.passport_number}
📅 Ngày sinh: {guest.dob}
⚥  Giới tính: {guest.gender}
🌍 Quốc gia cấp: {guest.issuing_country}
🏴 Quốc tịch: {guest.nationality}
📸 File: {guest.source_image}
🕒 Quét lúc: {guest.scan_time}
            """
            
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            self.info_text.config(state=tk.DISABLED)
            
            self.fill_btn.config(state=tk.NORMAL)
    
    def show_context_menu(self, event):
        """Right-click menu"""
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.context_menu.post(event.x_root, event.y_root)
    
    def on_double_click(self, event):
        """Double-click to copy"""
        region = self.tree.identify_region(event.x, event.y)
        if region == "cell":
            self.copy_selected_cell(event)
    
    def copy_selected_cell(self, event=None):
        """Copy cell"""
        try:
            selection = self.tree.selection()
            if not selection:
                return
            
            item = self.tree.item(selection[0])
            
            if event and hasattr(event, 'x'):
                column = self.tree.identify_column(event.x)
            else:
                column = "#1"
            
            col_map = {
                "#0": item['text'],
                "#1": item['values'][0] if len(item['values']) > 0 else "",
                "#2": item['values'][1] if len(item['values']) > 1 else "",
                "#3": item['values'][2] if len(item['values']) > 2 else "",
                "#4": item['values'][3] if len(item['values']) > 3 else "",
                "#5": item['values'][4] if len(item['values']) > 4 else "",
                "#6": item['values'][5] if len(item['values']) > 5 else "",
            }
            
            text_to_copy = str(col_map.get(column, ""))
            
            if text_to_copy:
                self.root.clipboard_clear()
                self.root.clipboard_append(text_to_copy)
                self.root.update()
                self.log(f"📋 Đã copy: {text_to_copy}")
        except Exception as e:
            self.log(f"❌ Lỗi copy: {e}")
    
    def copy_entire_row(self):
        """Copy toàn bộ dòng"""
        try:
            selection = self.tree.selection()
            if not selection:
                return
            
            item = self.tree.item(selection[0])
            index = int(item['text']) - 1
            
            if 0 <= index < len(self.guests):
                guest = self.guests[index]
                text = f"{guest.full_name}\t{guest.passport_number}\t{guest.dob}\t{guest.gender}\t{guest.issuing_country}\t{guest.nationality}"
                
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.root.update()
                
                self.log(f"📋 Đã copy toàn bộ dòng #{index + 1}")
        except Exception as e:
            self.log(f"❌ Lỗi copy: {e}")
    
    def fill_to_smile(self):
        """Điền vào Smile PMS"""
        selection = self.tree.selection()
        if not selection:
            return
        
        item = self.tree.item(selection[0])
        index = int(item['text']) - 1
        guest = self.guests[index]
        
        messagebox.showinfo("Thông báo", 
                           f"Chức năng điền vào Smile PMS\n"
                           f"Khách: {guest.full_name}\n"
                           f"(Sẽ được implement sau)")
        self.log(f"🔄 {guest.full_name} → Smile PMS (TODO)")
    
    def clear_all(self):
        """Xóa tất cả"""
        if not self.guests:
            return
        
        if messagebox.askyesno("Xác nhận", "Xóa tất cả khách đã quét?"):
            self.guests.clear()
            self.tree.delete(*self.tree.get_children())
            self.count_label.config(text="Tổng: 0 khách")
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete(1.0, tk.END)
            self.info_text.config(state=tk.DISABLED)
            self.fill_btn.config(state=tk.DISABLED)
            self.log("🗑️ Đã xóa tất cả")
    
    def log(self, message):
        """Ghi log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

# ============= MAIN =============
def main():
    root = TkinterDnD.Tk()
    app = MRZReaderApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
