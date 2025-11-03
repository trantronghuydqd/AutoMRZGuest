"""
GUI Application - Kéo thả ảnh + Lắng nghe thư mục để đọc MRZ với PaddleOCR (Độ chính xác cao)
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from tkinterdnd2 import DND_FILES, TkinterDnD
import threading
import os
from datetime import datetime
import re
import cv2
import numpy as np
from PIL import Image
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("⚠️ PaddleOCR chưa cài đặt. Chạy: pip install paddleocr")

# Fallback to PassportEye
try:
    from passporteye import read_mrz
    PASSPORTEYE_AVAILABLE = True
except ImportError:
    PASSPORTEYE_AVAILABLE = False

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

# ============= PADDLEOCR INITIALIZATION =============
if PADDLE_AVAILABLE:
    # Khởi tạo PaddleOCR (chỉ 1 lần)
    print("🚀 Đang khởi tạo PaddleOCR...")
    try:
        # Version mới của PaddleOCR
        paddle_ocr = PaddleOCR(
            use_textline_orientation=True,  # Tự động xoay (thay use_angle_cls)
            lang='en',                      # Ngôn ngữ tiếng Anh
            use_gpu=False                   # Dùng CPU
        )
        print("✅ PaddleOCR sẵn sàng!")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo PaddleOCR: {e}")
        print("🔄 Thử khởi tạo với config tối thiểu...")
        try:
            paddle_ocr = PaddleOCR(lang='en')
            print("✅ PaddleOCR sẵn sàng (config tối thiểu)!")
        except:
            paddle_ocr = None
            PADDLE_AVAILABLE = False
            print("❌ Không thể khởi tạo PaddleOCR")
else:
    paddle_ocr = None

# ============= IMAGE PREPROCESSING =============

def preprocess_for_mrz(image_path, output_dir=None):
    """Tiền xử lý ảnh tối ưu cho MRZ"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        height, width = img.shape[:2]
        
        # Xoay nếu ảnh dọc
        if height > width:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            height, width = img.shape[:2]
        
        # Crop vùng MRZ (20% dưới cùng để chắc chắn)
        mrz_region = img[int(height * 0.75):, :]
        
        # Convert to grayscale
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
        
        # Tăng kích thước 2x
        scale_factor = 2.0
        enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enlarged, None, h=10, 
                                           templateWindowSize=7, 
                                           searchWindowSize=21)
        
        # Adaptive threshold
        binary = cv2.adaptiveThreshold(denoised, 255, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Invert if needed
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Lưu ảnh preprocessed
        if output_dir and os.path.exists(output_dir):
            basename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(basename)[0]
            processed_path = os.path.join(output_dir, f"{name_without_ext}_paddle_prep.jpg")
        else:
            processed_path = image_path.rsplit('.', 1)[0] + '_paddle_prep.jpg'
        
        cv2.imwrite(processed_path, binary)
        
        return processed_path
        
    except Exception as e:
        print(f"Lỗi preprocess: {e}")
        return image_path

# ============= MRZ READER WITH PADDLEOCR =============

def fix_common_ocr_errors(text):
    """Sửa lỗi OCR phổ biến"""
    if not text:
        return ""
    
    corrections = {
        # S vs 5
        'I5HIKAWA': 'ISHIKAWA',
        'I5HII': 'ISHII',
        'TAKAHA5HI': 'TAKAHASHI',
        '5ATO': 'SATO',
        '5UZUKI': 'SUZUKI',
        '5HIMIZU': 'SHIMIZU',
        'KAWA5AKI': 'KAWASAKI',
        
        # I vs 1
        '1SHIKAWA': 'ISHIKAWA',
        '1SHII': 'ISHII',
        
        # O vs 0
        '0SAKA': 'OSAKA',
        'T0KY0': 'TOKYO',
        'YAMAM0T0': 'YAMAMOTO',
        
        # Common names
        'WATANAKE': 'WATANABE',
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def clean_name(name):
    """Làm sạch tên"""
    if not name:
        return ""
    
    name = fix_common_ocr_errors(name)
    name = name.replace('<<', '|SEP|')
    name = name.replace('<', '')
    
    parts = name.split('|SEP|')
    cleaned_parts = []
    
    for part in parts:
        cleaned = ''.join(c if c.isalpha() or c == ' ' else ' ' for c in part)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if cleaned:
            cleaned = cleaned.strip('K<')
            if cleaned:
                cleaned_parts.append(cleaned)
    
    result = ' '.join(cleaned_parts)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

def format_date_from_mrz(date_str):
    """Chuyển đổi YYMMDD -> dd/mm/yyyy"""
    if not date_str or len(date_str) != 6:
        return ""
    
    try:
        yy = int(date_str[:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        year = 2000 + yy if yy <= 30 else 1900 + yy
        return f"{dd:02d}/{mm:02d}/{year}"
    except:
        return date_str

def parse_mrz_lines(line1, line2):
    """Parse 2 dòng MRZ thủ công"""
    try:
        # Line 1: P<ISSUING_COUNTRY<<SURNAME<<GIVEN_NAMES
        # Line 2: PASSPORT_NUMBER<NATIONALITY<DOB<SEX<EXPIRY<...
        
        # Issuing country (vị trí 2-5 của line1)
        issuing = line1[2:5] if len(line1) > 5 else ""
        
        # Name (từ vị trí 5 đến hết line1)
        name_part = line1[5:44] if len(line1) >= 44 else line1[5:]
        parts = name_part.split('<<')
        surname = parts[0].replace('<', ' ').strip() if parts else ''
        given = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ''
        
        # Làm sạch tên
        surname = clean_name(surname)
        given = clean_name(given)
        full_name = f"{surname} {given}".strip()
        
        # Line 2 parsing
        passport = line2[0:9].replace('<', '').strip() if len(line2) >= 9 else ""
        nationality = line2[10:13] if len(line2) >= 13 else ""
        dob_raw = line2[13:19] if len(line2) >= 19 else ""
        sex = line2[20] if len(line2) > 20 else ""
        
        # Gender
        gender = 'M' if sex == 'M' else 'F' if sex == 'F' else ''
        
        return {
            'full_name': full_name,
            'passport_number': passport,
            'dob': format_date_from_mrz(dob_raw),
            'gender': gender,
            'issuing_country': issuing,
            'nationality': nationality
        }
    except Exception as e:
        print(f"Lỗi parse MRZ: {e}")
        return None

def read_mrz_with_paddle(image_path, output_dir=None):
    """Đọc MRZ bằng PaddleOCR"""
    if not PADDLE_AVAILABLE or paddle_ocr is None:
        print("❌ PaddleOCR không khả dụng")
        return None
    
    processed_path = None
    try:
        # Preprocess ảnh
        processed_path = preprocess_for_mrz(image_path, output_dir)
        
        # OCR với PaddleOCR
        result = paddle_ocr.ocr(processed_path, cls=True)
        
        if not result or not result[0]:
            return None
        
        # Extract text từ kết quả
        lines = []
        for line in result[0]:
            text = line[1][0]  # Get text
            confidence = line[1][1]  # Get confidence
            
            # Chỉ lấy dòng có độ tin cậy cao và có dấu <
            if confidence > 0.5 and '<' in text:
                # Clean text
                clean_text = text.replace(' ', '').replace('|', 'I').replace('O', '0')
                lines.append(clean_text)
        
        print(f"📝 PaddleOCR tìm thấy {len(lines)} dòng MRZ")
        for i, line in enumerate(lines, 1):
            print(f"   Line {i}: {line}")
        
        # Tìm 2 dòng MRZ hợp lệ (độ dài ~44 ký tự)
        valid_lines = [l for l in lines if 40 <= len(l) <= 50]
        
        if len(valid_lines) >= 2:
            line1, line2 = valid_lines[0], valid_lines[1]
            parsed = parse_mrz_lines(line1, line2)
            
            if parsed:
                print(f"✅ PaddleOCR đọc thành công: {parsed['full_name']}")
                return parsed
        
        return None
        
    except Exception as e:
        print(f"❌ Lỗi PaddleOCR: {e}")
        return None
    
    finally:
        # Chỉ xóa file preprocessed nếu KHÔNG có output_dir (tức là kéo thả)
        # Nếu có output_dir (lắng nghe thư mục) thì GIỮ LẠI file preprocessed
        if not output_dir:
            if processed_path and processed_path != image_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except:
                    pass  # Silent fail

def format_date_from_string(date_str):
    """Chuyển đổi ngày từ nhiều format về dd/mm/yyyy"""
    if not date_str:
        return ""
    
    # Nếu đã đúng format dd/mm/yyyy
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                return date_str  # Đã đúng dd/mm/yyyy
            if len(parts[0]) == 4:
                return f"{parts[2]}/{parts[1]}/{parts[0]}"  # yyyy/mm/dd -> dd/mm/yyyy
    
    # Nếu là YYYY-MM-DD
    if '-' in date_str and len(date_str) == 10:
        parts = date_str.split('-')
        if len(parts[0]) == 4:
            return f"{parts[2]}/{parts[1]}/{parts[0]}"  # YYYY-MM-DD -> dd/mm/yyyy
    
    # Nếu là YYMMDD (6 số)
    if len(date_str) == 6 and date_str.isdigit():
        return format_date_from_mrz(date_str)
    
    return date_str

def read_mrz_with_passporteye(image_path):
    """Fallback: Đọc MRZ bằng PassportEye"""
    if not PASSPORTEYE_AVAILABLE:
        return None
    
    try:
        print("🔄 Thử PassportEye...")
        mrz_obj = read_mrz(image_path)
        
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
        
        # Format ngày sinh về dd/mm/yyyy
        dob = format_date_from_string(mrz_data.get('date_of_birth', ''))
        
        return {
            'full_name': full_name,
            'passport_number': mrz_data.get('number', ''),
            'dob': dob,
            'gender': gender,
            'issuing_country': mrz_data.get('country', ''),
            'nationality': mrz_data.get('nationality', '')
        }
    except Exception as e:
        print(f"❌ Lỗi PassportEye: {e}")
        return None

def read_mrz_from_image(image_path, output_dir=None):
    """Đọc MRZ - Ưu tiên PaddleOCR, fallback PassportEye"""
    
    # Strategy 1: PaddleOCR (chính xác nhất)
    result = read_mrz_with_paddle(image_path, output_dir)
    
    # Strategy 2: PassportEye (backup)
    if not result:
        result = read_mrz_with_passporteye(image_path)
    
    if not result:
        return None
    
    # Convert dict to Guest object
    try:
        guest = Guest(
            full_name=result['full_name'],
            passport_number=result['passport_number'],
            dob=result['dob'],
            gender=result['gender'],
            issuing_country=result['issuing_country'],
            nationality=result['nationality'],
            source_image=os.path.basename(image_path)
        )
        return guest
    except Exception as e:
        print(f"Lỗi tạo Guest: {e}")
        return None

# ============= FOLDER WATCHER =============
class ImageFolderHandler(FileSystemEventHandler):
    """Xử lý sự kiện khi có file mới trong thư mục"""
    def __init__(self, app):
        self.app = app
        self.processed_files = set()  # Tránh xử lý trùng
    
    def on_created(self, event):
        """Khi có file mới được tạo"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Chỉ xử lý file ảnh
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return
        
        # Tránh xử lý file preprocessed
        if '_paddle_prep.jpg' in file_path:
            return
        
        # Tránh xử lý trùng
        if file_path in self.processed_files:
            return
        
        self.processed_files.add(file_path)
        
        # Đợi file được ghi xong (tránh lỗi khi file đang copy)
        time.sleep(0.5)
        
        if os.path.exists(file_path):
            self.app.log(f"🔔 Phát hiện ảnh mới: {os.path.basename(file_path)}")
            threading.Thread(target=self.app.process_images, 
                           args=([file_path],), 
                           daemon=True).start()

# ============= GUI APPLICATION =============
class MRZReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 MRZ Reader - PaddleOCR Edition + Folder Watcher")
        self.root.geometry("1200x800")
        
        self.guests = []
        self.processing = False
        
        # Folder watcher
        self.watch_folder = ""
        self.preprocessed_folder = ""
        self.observer = None
        self.watching = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Tạo giao diện"""
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=80)
        header.pack(fill=tk.X)
        
        ocr_status = "🚀 PaddleOCR" if PADDLE_AVAILABLE else "⚠️ PassportEye Only"
        title = tk.Label(header, text=f"📖 MRZ READER - {ocr_status} + 👁️ FOLDER WATCHER", 
                        font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
        title.pack(pady=5)
        
        subtitle = tk.Label(header, text="Kéo thả ảnh HOẶC lắng nghe thư mục tự động", 
                           font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1")
        subtitle.pack(pady=2)
        
        # Folder Watcher Control Panel
        watcher_frame = tk.Frame(self.root, bg="#34495e", height=100)
        watcher_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Row 1: Watch Folder
        row1 = tk.Frame(watcher_frame, bg="#34495e")
        row1.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row1, text="📂 Thư mục lắng nghe:", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white", width=18, anchor='w').pack(side=tk.LEFT)
        
        self.watch_folder_var = tk.StringVar(value="Chưa chọn")
        tk.Label(row1, textvariable=self.watch_folder_var, font=("Arial", 9), 
                bg="#2c3e50", fg="white", anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Button(row1, text="Chọn thư mục", command=self.select_watch_folder,
                 bg="#3498db", fg="white", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=2)
        
        # Row 2: Preprocessed Folder
        row2 = tk.Frame(watcher_frame, bg="#34495e")
        row2.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row2, text="💾 Thư mục preprocessed:", font=("Arial", 10, "bold"), 
                bg="#34495e", fg="white", width=18, anchor='w').pack(side=tk.LEFT)
        
        self.preprocessed_folder_var = tk.StringVar(value="Chưa chọn")
        tk.Label(row2, textvariable=self.preprocessed_folder_var, font=("Arial", 9), 
                bg="#2c3e50", fg="white", anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Button(row2, text="Chọn thư mục", command=self.select_preprocessed_folder,
                 bg="#3498db", fg="white", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=2)
        
        # Row 3: Control Buttons
        row3 = tk.Frame(watcher_frame, bg="#34495e")
        row3.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_watch_btn = tk.Button(row3, text="▶️ BẮT ĐẦU QUÉT", 
                                         command=self.start_watching,
                                         bg="#27ae60", fg="white", 
                                         font=("Arial", 11, "bold"), height=1, width=20)
        self.start_watch_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_watch_btn = tk.Button(row3, text="⏸️ DỪNG QUÉT", 
                                        command=self.stop_watching,
                                        bg="#e74c3c", fg="white", 
                                        font=("Arial", 11, "bold"), height=1, width=20,
                                        state=tk.DISABLED)
        self.stop_watch_btn.pack(side=tk.LEFT, padx=5)
        
        self.watch_status_label = tk.Label(row3, text="⏸️ Chưa quét", 
                                           font=("Arial", 10, "bold"), 
                                           bg="#34495e", fg="#ecf0f1")
        self.watch_status_label.pack(side=tk.LEFT, padx=20)
        
        # Main container
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Drop Zone & Guest List
        left_frame = tk.Frame(main)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Drop zone
        drop_frame = tk.LabelFrame(left_frame, text="📥 KÉO THẢ ẢNH VÀO ĐÂY", 
                                   font=("Arial", 12, "bold"), bg="#ecf0f1", height=120)
        drop_frame.pack(fill=tk.X, pady=(0, 10))
        drop_frame.pack_propagate(False)
        
        self.drop_label = tk.Label(drop_frame, 
                                   text="🖼️ Kéo thả ảnh passport (PaddleOCR - Độ chính xác 90%+)\n(JPG, PNG, JPEG)",
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
        if PADDLE_AVAILABLE:
            self.log("🚀 PaddleOCR đã sẵn sàng (Độ chính xác 90%+)")
        else:
            self.log("⚠️ PaddleOCR chưa cài - Dùng PassportEye")
            self.log("💡 Cài PaddleOCR: pip install paddleocr")
    
    def on_drop(self, event):
        """Xử lý khi kéo thả file"""
        if self.processing:
            self.log("⚠️ Đang xử lý, vui lòng đợi...")
            return
        
        files = self.root.tk.splitlist(event.data)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            self.log("❌ Không có file ảnh hợp lệ")
            return
        
        self.log(f"📥 Nhận {len(image_files)} ảnh")
        
        threading.Thread(target=self.process_images, args=(image_files,), daemon=True).start()
    
    def select_watch_folder(self):
        """Chọn thư mục lắng nghe"""
        folder = filedialog.askdirectory(title="Chọn thư mục lắng nghe")
        if folder:
            self.watch_folder = folder
            self.watch_folder_var.set(folder)
            self.log(f"📂 Đã chọn thư mục lắng nghe: {folder}")
    
    def select_preprocessed_folder(self):
        """Chọn thư mục lưu preprocessed"""
        folder = filedialog.askdirectory(title="Chọn thư mục lưu preprocessed")
        if folder:
            self.preprocessed_folder = folder
            self.preprocessed_folder_var.set(folder)
            self.log(f"💾 Đã chọn thư mục preprocessed: {folder}")
    
    def start_watching(self):
        """Bắt đầu lắng nghe thư mục"""
        if not self.watch_folder or not os.path.exists(self.watch_folder):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục lắng nghe!")
            return
        
        if not self.preprocessed_folder or not os.path.exists(self.preprocessed_folder):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục lưu preprocessed!")
            return
        
        try:
            # Tạo observer
            event_handler = ImageFolderHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.watch_folder, recursive=False)
            self.observer.start()
            
            self.watching = True
            self.watch_status_label.config(text="✅ Đang quét...", fg="#2ecc71")
            self.start_watch_btn.config(state=tk.DISABLED)
            self.stop_watch_btn.config(state=tk.NORMAL)
            
            self.log(f"👁️ Bắt đầu lắng nghe: {self.watch_folder}")
            self.log(f"💾 File preprocessed sẽ lưu tại: {self.preprocessed_folder}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể bắt đầu lắng nghe:\n{e}")
            self.log(f"❌ Lỗi bắt đầu lắng nghe: {e}")
    
    def stop_watching(self):
        """Dừng lắng nghe thư mục"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
        
        self.watching = False
        self.watch_status_label.config(text="⏸️ Đã dừng", fg="#95a5a6")
        self.start_watch_btn.config(state=tk.NORMAL)
        self.stop_watch_btn.config(state=tk.DISABLED)
        
        self.log("⏸️ Đã dừng lắng nghe thư mục")
    
    def process_images(self, image_files):
        """Xử lý nhiều ảnh"""
        self.processing = True
        self.status_label.config(text="⏳ Đang xử lý...", fg="orange")
        
        for image_path in image_files:
            try:
                self.log(f"📸 Đọc: {os.path.basename(image_path)}")
                
                # Nếu đang watching, truyền preprocessed_folder
                output_dir = self.preprocessed_folder if self.watching else None
                guest = read_mrz_from_image(image_path, output_dir)
                
                if guest:
                    self.add_guest(guest)
                    self.log(f"✅ {guest.full_name} - {guest.passport_number}")
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
    
    def on_closing(self):
        """Xử lý khi đóng app"""
        if self.watching:
            if messagebox.askyesno("Xác nhận", "Đang lắng nghe thư mục. Bạn có muốn dừng và thoát?"):
                self.stop_watching()
                self.root.destroy()
        else:
            self.root.destroy()
    
    def log(self, message):
        """Ghi log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

# ============= MAIN =============
def main():
    if not PADDLE_AVAILABLE:
        print("\n" + "="*60)
        print("⚠️  CẢNH BÁO: PaddleOCR chưa được cài đặt!")
        print("="*60)
        print("PaddleOCR giúp tăng độ chính xác lên 90%+")
        print("\nCài đặt:")
        print("  pip install paddleocr")
        print("  pip install paddlepaddle")
        print("\nỨng dụng sẽ dùng PassportEye thay thế (độ chính xác thấp hơn)")
        print("="*60 + "\n")
    
    print("\n" + "="*60)
    print("📦 Kiểm tra thư viện watchdog...")
    print("="*60)
    try:
        from watchdog.observers import Observer
        print("✅ watchdog đã sẵn sàng")
    except ImportError:
        print("⚠️  watchdog chưa cài đặt!")
        print("Cài đặt: pip install watchdog")
    print("="*60 + "\n")
    
    root = TkinterDnD.Tk()
    app = MRZReaderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
