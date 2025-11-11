
"""
GUI Application - Kéo thả ảnh + Lắng nghe thư mục để đọc MRZ
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
from PIL import Image, ImageTk
import json
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Tắt warnings không cần thiết
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from passporteye import read_mrz

# File config để lưu đường dẫn
CONFIG_FILE = "mrz_config.json"

# ============= CONFIG MANAGER =============
class ConfigManager:
    """Quản lý config - Lưu/Load đường dẫn"""
    
    @staticmethod
    def load_config():
        """Load config từ file"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Lỗi load config: {e}")
        
        return {
            'watch_folder': '',
            'process_folder': ''
        }
    
    @staticmethod
    def save_config(watch_folder, process_folder):
        """Lưu config vào file"""
        try:
            config = {
                'watch_folder': watch_folder,
                'process_folder': process_folder
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu config: {CONFIG_FILE}")
        except Exception as e:
            print(f"Lỗi save config: {e}")

# ============= FOLDER WATCHER =============
class ImageFolderHandler(FileSystemEventHandler):
    """Xử lý sự kiện khi có file mới trong thư mục"""
    def __init__(self, app):
        self.app = app
        self.processed_files = set()
    
    def on_created(self, event):
        """Khi có file mới được tạo"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        
        # Chỉ xử lý file ảnh
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            return
        
        # Tránh xử lý file tạm
        if '_rotated' in file_path or '_enhanced' in file_path:
            return
        
        # Tránh xử lý trùng
        if file_path in self.processed_files:
            return
        
        self.processed_files.add(file_path)
        
        # Đợi file được ghi xong
        time.sleep(0.5)
        
        if os.path.exists(file_path):
            self.app.log(f"🔔 Phát hiện ảnh mới: {os.path.basename(file_path)}")
            threading.Thread(target=self.app.process_images, 
                           args=([file_path],), 
                           daemon=True).start()

# ============= GUEST MODEL (OOP) =============
class Guest:
    """Object lưu thông tin khách (giống OOP Java) - BỎ expiry_date"""
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

def enhance_mrz_region(image_path):
    """
    THUẬT TOÁN XỬ LÝ ẢNH THÔNG MINH:
    Tăng độ chính xác OCR cho vùng MRZ bằng cách:
    1. Crop chỉ vùng MRZ (25% dưới cùng)
    2. Tăng kích thước 3x (làm chữ to hơn)
    3. Denoise (khử nhiễu)
    4. Tăng độ tương phản (CLAHE)
    5. Binary threshold (chỉ giữ chữ đen/trắng)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        height, width = img.shape[:2]
        
        # Xoay nếu ảnh dọc
        if height > width:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            height, width = img.shape[:2]
        
        # Bước 1: Crop vùng MRZ (25% dưới cùng)
        mrz_height = int(height * 0.25)
        mrz_region = img[height - mrz_height:, :]
        
        # Bước 2: Convert sang grayscale
        gray = cv2.cvtColor(mrz_region, cv2.COLOR_BGR2GRAY)
        
        # Bước 3: Tăng kích thước 3x (làm chữ to, dễ nhận diện)
        scale_factor = 3.0
        enlarged = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_CUBIC)
        
        # Bước 4: Denoise (khử nhiễu background)
        denoised = cv2.fastNlMeansDenoising(enlarged, None, h=10, 
                                           templateWindowSize=7, 
                                           searchWindowSize=21)
        
        # Bước 5: Tăng độ tương phản bằng CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(denoised)
        
        # Bước 6: Binary threshold (chỉ giữ đen/trắng)
        # Dùng Otsu để tự động tìm threshold tối ưu
        _, binary = cv2.threshold(contrasted, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Bước 7: Morphology để làm sạch chữ
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Bước 8: Đảo màu nếu background là đen
        if np.mean(cleaned) < 127:
            cleaned = cv2.bitwise_not(cleaned)
        
        # Lưu ảnh đã xử lý
        enhanced_path = image_path.rsplit('.', 1)[0] + '_enhanced.jpg'
        cv2.imwrite(enhanced_path, cleaned)
        
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

# ============= MRZ READER =============
def fix_ocr_errors_smart(text):
    """
    THUẬT TOÁN SỬA LỖI OCR THÔNG MINH:
    Không dùng dictionary cứng, mà dùng PATTERN MATCHING
    
    Nguyên tắc:
    1. Số 0 ở giữa/cuối từ → chuyển thành O
    2. Số 1 ở đầu từ → chuyển thành I
    3. Số 5 ở đầu từ → chuyển thành S
    4. Ký tự đơn lẻ K, <, | ở cuối → xóa
    """
    if not text:
        return ""
    
    # Split thành từng từ (họ và tên riêng biệt)
    words = text.split()
    fixed_words = []
    
    for word in words:
        if not word:
            continue
            
        # Chuyển thành list để dễ sửa từng ký tự
        chars = list(word)
        
        # Rule 1: Số 0 ở giữa hoặc cuối từ → O
        # VD: TAR0 → TARO, T0M → TOM
        for i in range(len(chars)):
            if chars[i] == '0':
                # Nếu có chữ cái trước và sau, hoặc ở cuối
                if i > 0 and chars[i-1].isalpha():
                    chars[i] = 'O'
        
        # Rule 2: Số 1 ở đầu hoặc giữa từ → I
        # VD: 1AN → IAN, KEN1 → KENI
        for i in range(len(chars)):
            if chars[i] == '1':
                if i == 0 or (i > 0 and chars[i-1].isalpha()):
                    chars[i] = 'I'
        
        # Rule 3: Số 5 ở đầu từ → S
        # VD: 5ATO → SATO, 5MITH → SMITH
        if len(chars) > 0 and chars[0] == '5':
            chars[0] = 'S'
        
        # Rule 4: Số 5 ở giữa/cuối sau nguyên âm → S
        # VD: MA5AYA → MASAYA
        vowels = 'AEIOU'
        for i in range(1, len(chars)):
            if chars[i] == '5' and i > 0 and chars[i-1] in vowels:
                chars[i] = 'S'
        
        # Rule 5: Số 3 giữa/cuối → E
        # VD: TYL3R → TYLER
        for i in range(1, len(chars)):
            if chars[i] == '3':
                chars[i] = 'E'
        
        # Rule 6: Số 8 → B
        # VD: 8EN → BEN
        for i in range(len(chars)):
            if chars[i] == '8':
                chars[i] = 'B'
        
        fixed_word = ''.join(chars)
        
        # Rule 7: Xóa ký tự đơn lẻ ở cuối (K, <, |)
        fixed_word = fixed_word.rstrip('K<|')
        
        if fixed_word:
            fixed_words.append(fixed_word)
    
    return ' '.join(fixed_words)

def clean_name(name):
    """
    THUẬT TOÁN LÀM SẠCH TÊN THÔNG MINH:
    Không dùng dictionary cứng, dùng pattern matching
    """
    if not name:
        return ""
    
    # Bước 1: Xử lý separator << (giữ lại để tách họ và tên)
    name = name.replace('<<', '|SEP|')
    name = name.replace('<', ' ')
    
    # Bước 2: Tách thành họ và tên
    parts = name.split('|SEP|')
    cleaned_parts = []
    
    for part in parts:
        # Loại bỏ ký tự đặc biệt, chỉ giữ chữ cái, số, space
        temp = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in part)
        temp = re.sub(r'\s+', ' ', temp).strip()
        
        if temp:
            # Áp dụng THUẬT TOÁN sửa lỗi OCR thông minh
            fixed = fix_ocr_errors_smart(temp)
            
            # Xóa ký tự thừa ở đầu/cuối
            fixed = fixed.strip('K<| ')
            
            if fixed:
                cleaned_parts.append(fixed)
    
    # Bước 3: Ghép lại
    result = ' '.join(cleaned_parts)
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

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

def read_mrz_from_image(image_path):
    """Đọc MRZ và trả về Guest object - CHIẾN LƯỢC 2 LẦN ĐỌC"""
    try:
        # Bước 1: Xoay ảnh nếu cần
        rotated_path = rotate_image_if_needed(image_path)
        
        # CHIẾN LƯỢC 1: Thử đọc từ ảnh gốc (hoặc đã xoay) trước
        print("🔄 Thử đọc từ ảnh gốc...")
        mrz_obj = read_mrz(rotated_path)
        
        # CHIẾN LƯỢC 2: Nếu thất bại, thử với ảnh đã enhance
        if not mrz_obj:
            print("🔄 Thử đọc từ ảnh enhanced...")
            enhanced_path = enhance_mrz_region(rotated_path)
            mrz_obj = read_mrz(enhanced_path)
            
            # Xóa file enhanced
            if enhanced_path != rotated_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)
        
        # Xóa file rotated
        if rotated_path != image_path and os.path.exists(rotated_path):
            os.remove(rotated_path)
        
        if not mrz_obj:
            print("❌ Không đọc được MRZ từ cả 2 phương pháp")
            return None
        
        print("✅ Đọc MRZ thành công!")
        
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
            source_image=os.path.basename(image_path)
        )
        
        return guest
    except Exception as e:
        print(f"Lỗi đọc MRZ: {e}")
        return None

# ============= GUI APPLICATION =============
class MRZReaderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧩 MRZ Reader - Drag & Drop + Folder Watcher")
        self.root.geometry("1400x850")
        
        self.guests = []
        self.processing = False
        
        # Folder watcher
        self.watch_folder = ""
        self.process_folder = ""
        self.observer = None
        self.watching = False
        
        # Load config
        self.load_saved_config()
        
        self.setup_ui()
        
        # Auto-start watching nếu có config
        if self.watch_folder and self.process_folder:
            self.root.after(500, self.start_watching)
    
    def load_saved_config(self):
        """Load config đã lưu"""
        config = ConfigManager.load_config()
        self.watch_folder = config.get('watch_folder', '')
        self.process_folder = config.get('process_folder', '')
    
    def setup_ui(self):
        """Tạo giao diện"""
        # Header
        header = tk.Frame(self.root, bg="#2c3e50", height=80)
        header.pack(fill=tk.X)
        
        title = tk.Label(header, text="📖 MRZ READER - DRAG & DROP + 👁️ FOLDER WATCHER", 
                        font=("Arial", 16, "bold"), bg="#2c3e50", fg="white")
        title.pack(pady=5)
        
        subtitle = tk.Label(header, text="Kéo thả ảnh HOẶC lắng nghe thư mục tự động", 
                           font=("Arial", 10), bg="#2c3e50", fg="#ecf0f1")
        subtitle.pack(pady=2)
        
        # Folder Watcher Control Panel
        watcher_frame = tk.Frame(self.root, bg="#34495e", height=120)
        watcher_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Row 1: Watch Folder
        row1 = tk.Frame(watcher_frame, bg="#34495e")
        row1.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row1, text="📂 Thư mục lắng nghe:", font=("Arial", 9, "bold"), 
                bg="#34495e", fg="white", width=16, anchor='w').pack(side=tk.LEFT)
        
        self.watch_folder_var = tk.StringVar(value=self.watch_folder or "Chưa chọn")
        tk.Label(row1, textvariable=self.watch_folder_var, font=("Arial", 8), 
                bg="#2c3e50", fg="white", anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Button(row1, text="Chọn", command=self.select_watch_folder,
                 bg="#3498db", fg="white", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=2)
        
        # Row 2: Process Folder
        row2 = tk.Frame(watcher_frame, bg="#34495e")
        row2.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(row2, text="💾 Thư mục đã xử lý:", font=("Arial", 9, "bold"), 
                bg="#34495e", fg="white", width=16, anchor='w').pack(side=tk.LEFT)
        
        self.process_folder_var = tk.StringVar(value=self.process_folder or "Chưa chọn")
        tk.Label(row2, textvariable=self.process_folder_var, font=("Arial", 8), 
                bg="#2c3e50", fg="white", anchor='w').pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        tk.Button(row2, text="Chọn", command=self.select_process_folder,
                 bg="#3498db", fg="white", font=("Arial", 8, "bold")).pack(side=tk.RIGHT, padx=2)
        
        # Row 3: Control Buttons
        row3 = tk.Frame(watcher_frame, bg="#34495e")
        row3.pack(fill=tk.X, padx=10, pady=5)
        
        self.start_watch_btn = tk.Button(row3, text="▶️ BẮT ĐẦU QUÉT", 
                                         command=self.start_watching,
                                         bg="#27ae60", fg="white", 
                                         font=("Arial", 10, "bold"), height=1, width=18)
        self.start_watch_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_watch_btn = tk.Button(row3, text="⏸️ DỪNG QUÉT", 
                                        command=self.stop_watching,
                                        bg="#e74c3c", fg="white", 
                                        font=("Arial", 10, "bold"), height=1, width=18,
                                        state=tk.DISABLED)
        self.stop_watch_btn.pack(side=tk.LEFT, padx=5)
        
        self.watch_status_label = tk.Label(row3, text="⏸️ Chưa quét", 
                                           font=("Arial", 9, "bold"), 
                                           bg="#34495e", fg="#ecf0f1")
        self.watch_status_label.pack(side=tk.LEFT, padx=15)
        
        self.scan_folder_btn = tk.Button(row3, text="🔍 QUÉT THƯ MỤC", 
                                         command=self.scan_folder_images,
                                         bg="#9b59b6", fg="white", 
                                         font=("Arial", 10, "bold"), height=1, width=18)
        self.scan_folder_btn.pack(side=tk.LEFT, padx=5)
        
        # Main container
        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Drop Zone & Guest List
        left_frame = tk.Frame(main)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Drop zone
        drop_frame = tk.LabelFrame(left_frame, text="📥 KÉO THẢ ẢNH VÀO ĐÂY", 
                                   font=("Arial", 11, "bold"), bg="#ecf0f1", height=90)
        drop_frame.pack(fill=tk.X, pady=(0, 5))
        drop_frame.pack_propagate(False)
        
        self.drop_label = tk.Label(drop_frame, 
                                   text="🖼️ Kéo thả 1 hoặc nhiều ảnh passport vào đây\n(JPG, PNG, JPEG)",
                                   font=("Arial", 11), bg="#ecf0f1", fg="#7f8c8d")
        self.drop_label.pack(expand=True)
        
        # Enable drag & drop
        drop_frame.drop_target_register(DND_FILES)
        drop_frame.dnd_bind('<<Drop>>', self.on_drop)
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
        
        # Guest list - BỎ cột Expiry
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
        self.log("💡 Kéo thả ảnh passport vào khung phía trên")
    
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
                
                guest = read_mrz_from_image(image_path)
                
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
        """Thêm guest vào list - BỎ expiry_date"""
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
        """Khi chọn guest - BỎ expiry_date"""
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
        """Copy cell - CẬP NHẬT column map"""
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
                "#1": item['values'][0] if len(item['values']) > 0 else "",  # Name
                "#2": item['values'][1] if len(item['values']) > 1 else "",  # Passport
                "#3": item['values'][2] if len(item['values']) > 2 else "",  # DOB
                "#4": item['values'][3] if len(item['values']) > 3 else "",  # Gender
                "#5": item['values'][4] if len(item['values']) > 4 else "",  # Issuing
                "#6": item['values'][5] if len(item['values']) > 5 else "",  # Nationality
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
        """Copy toàn bộ dòng - BỎ expiry_date"""
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
    
    def select_watch_folder(self):
        """Chọn thư mục lắng nghe"""
        folder = filedialog.askdirectory(title="Chọn thư mục lắng nghe")
        if folder:
            self.watch_folder = folder
            self.watch_folder_var.set(folder)
            self.log(f"📂 Đã chọn thư mục lắng nghe: {folder}")
            ConfigManager.save_config(self.watch_folder, self.process_folder)
    
    def select_process_folder(self):
        """Chọn thư mục đã xử lý"""
        folder = filedialog.askdirectory(title="Chọn thư mục đã xử lý")
        if folder:
            self.process_folder = folder
            self.process_folder_var.set(folder)
            self.log(f"💾 Đã chọn thư mục đã xử lý: {folder}")
            ConfigManager.save_config(self.watch_folder, self.process_folder)
    
    def start_watching(self):
        """Bắt đầu lắng nghe thư mục"""
        if not self.watch_folder or not os.path.exists(self.watch_folder):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục lắng nghe!")
            return
        
        if not self.process_folder or not os.path.exists(self.process_folder):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục đã xử lý!")
            return
        
        try:
            event_handler = ImageFolderHandler(self)
            self.observer = Observer()
            self.observer.schedule(event_handler, self.watch_folder, recursive=False)
            self.observer.start()
            
            self.watching = True
            self.watch_status_label.config(text="✅ Đang quét...", fg="#2ecc71")
            self.start_watch_btn.config(state=tk.DISABLED)
            self.stop_watch_btn.config(state=tk.NORMAL)
            
            self.log(f"👁️ Bắt đầu lắng nghe: {self.watch_folder}")
            self.log(f"💾 File đã xử lý sẽ chuyển đến: {self.process_folder}")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể bắt đầu lắng nghe:\n{e}")
            self.log(f"❌ Lỗi: {e}")
    
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
    
    def scan_folder_images(self):
        """Quét tất cả ảnh trong thư mục"""
        if not self.watch_folder or not os.path.exists(self.watch_folder):
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn thư mục lắng nghe trước!")
            return
        
        try:
            image_files = []
            for filename in os.listdir(self.watch_folder):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    if '_rotated' not in filename and '_enhanced' not in filename:
                        image_files.append(os.path.join(self.watch_folder, filename))
            
            if image_files:
                self.log(f"🔍 Tìm thấy {len(image_files)} ảnh trong thư mục")
                threading.Thread(target=self.process_images, args=(image_files,), daemon=True).start()
            else:
                self.log("⚠️ Không tìm thấy ảnh trong thư mục")
        except Exception as e:
            self.log(f"❌ Lỗi quét thư mục: {e}")
    
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
    root = TkinterDnD.Tk()
    app = MRZReaderApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
