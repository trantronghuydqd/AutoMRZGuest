"""
Script đơn giản đọc MRZ từ file ảnh JPG
Sử dụng cả PassportEye và Tesseract OCR
"""
import sys
import json
from passporteye import read_mrz
import pytesseract
import cv2
import numpy as np
import re

def format_date(yymmdd):
    """Chuyển YYMMDD -> dd/mm/yyyy"""
    if not yymmdd or len(yymmdd) != 6:
        return ""
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
    year = 2000 + yy if yy <= 30 else 1900 + yy
    return f"{dd:02d}/{mm:02d}/{year}"

def format_date_from_string(date_str):
    """Chuyển đổi ngày từ nhiều format về dd/mm/yyyy - FIX"""
    if not date_str:
        return ""
    
    # Nếu đã đúng format dd/mm/yyyy
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            # Kiểm tra xem có phải dd/mm/yyyy không
            if len(parts[0]) <= 2 and len(parts[1]) <= 2 and len(parts[2]) == 4:
                return date_str  # Đã đúng dd/mm/yyyy
            # Nếu là yyyy/mm/dd
            if len(parts[0]) == 4:
                return f"{parts[2]}/{parts[1]}/{parts[0]}"
    
    # Nếu là YYYY-MM-DD
    if '-' in date_str and len(date_str) == 10:
        parts = date_str.split('-')
        if len(parts[0]) == 4:  # YYYY-MM-DD
            return f"{parts[2]}/{parts[1]}/{parts[0]}"  # dd/mm/yyyy
    
    # Nếu là YYMMDD
    if len(date_str) == 6 and date_str.isdigit():
        return format_date(date_str)
    
    return date_str

def clean_name(name):
    """Làm sạch tên: loại bỏ ký tự lặp lại và < thừa"""
    if not name:
        return ""
    
    # Loại bỏ dấu < và chuyển K thành khoảng trắng (PassportEye đọc nhầm < thành K)
    name = name.replace('<', ' ').replace('K', ' ')
    
    # Loại bỏ khoảng trắng thừa và các ký tự lặp lại
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Loại bỏ chuỗi ký tự lặp 3 lần trở lên (KKKKKK, <<<<<, etc)
    name = re.sub(r'(.)\1{2,}', '', name)
    
    return name.strip()

def read_with_passporteye(image_path):
    """Phương pháp 1: PassportEye (nhanh & chính xác)"""
    print("\n🚀 [PassportEye] Đang đọc MRZ...")
    try:
        mrz_obj = read_mrz(image_path)
        if not mrz_obj:
            print("❌ [PassportEye] Không tìm thấy MRZ")
            return None
        
        mrz_data = mrz_obj.to_dict()
        
        if not mrz_data:
            print("❌ [PassportEye] MRZ data rỗng")
            return None
        
        print(f"✅ [PassportEye] Tìm thấy MRZ")
        
        # Làm sạch tên
        surname = clean_name(mrz_data.get('surname', ''))
        given_names = clean_name(mrz_data.get('names', ''))
        full_name = f"{surname} {given_names}".strip()
        
        # Lấy giới tính: M hoặc F
        sex = mrz_data.get('sex', '')
        gender = 'M' if sex == 'M' else 'F' if sex == 'F' else ''
        
        # Chỉ giữ 7 trường cần thiết (thêm gender)
        result = {
            "full_name": full_name,
            "passport_number": mrz_data.get('number', ''),
            "dob": format_date_from_string(mrz_data.get('date_of_birth', '')),
            "gender": gender,
            "issuing_country": mrz_data.get('country', ''),
            "nationality": mrz_data.get('nationality', ''),
            "expiry_date": format_date_from_string(mrz_data.get('expiration_date', ''))
        }
        
        print(f"✅ [PassportEye] Parsed:")
        print(f"   Tên: {result['full_name']}")
        print(f"   Passport: {result['passport_number']}")
        print(f"   Ngày sinh: {result['dob']}")
        print(f"   Giới tính: {result['gender']}")
        print(f"   Quốc gia cấp: {result['issuing_country']}")
        print(f"   Quốc tịch: {result['nationality']}")
        print(f"   Ngày hết hạn: {result['expiry_date']}")
        
        return result
        
    except Exception as e:
        print(f"❌ [PassportEye] Lỗi: {e}")
        return None

def preprocess_mrz_region(image):
    """Tiền xử lý ảnh để cải thiện OCR"""
    # Resize nếu quá nhỏ
    height, width = image.shape[:2]
    if height < 200:
        scale = 200 / height
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Tăng contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 30, 7, 21)
    
    # Threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphology để làm rõ chữ
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return morph

def read_with_tesseract(image_path):
    """Phương pháp 2: Tesseract OCR (backup)"""
    print("\n📖 [Tesseract] Đang đọc MRZ...")
    try:
        # Đọc ảnh
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        print(f"   Kích thước ảnh: {width}x{height}")
        
        # Thử nhiều vùng crop khác nhau
        crop_percentages = [0.85, 0.80, 0.75]  # Crop từ 15%, 20%, 25% dưới cùng
        
        all_lines = []
        
        for crop_pct in crop_percentages:
            print(f"\n   Thử crop từ {int((1-crop_pct)*100)}% dưới cùng...")
            
            # Crop vùng MRZ
            mrz_region = img[int(height * crop_pct):, :]
            
            # Preprocess
            processed = preprocess_mrz_region(mrz_region)
            
            # Lưu ảnh debug
            debug_file = f"debug_tesseract_{int((1-crop_pct)*100)}.jpg"
            cv2.imwrite(debug_file, processed)
            print(f"   Đã lưu: {debug_file}")
            
            # OCR với config tối ưu
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
            text = pytesseract.image_to_string(processed, config=custom_config)
            
            # Làm sạch và tìm dòng MRZ
            for line in text.split('\n'):
                cleaned = line.strip().replace(' ', '').replace('|', 'I').replace('O', '0')
                # MRZ phải có độ dài 44 ký tự
                if len(cleaned) == 44 and ('<' in cleaned or cleaned[:2] in ['P<', 'V<', 'I<']):
                    print(f"   ✅ Tìm thấy dòng: {cleaned}")
                    all_lines.append(cleaned)
        
        # Lọc lấy 2 dòng tốt nhất
        unique_lines = []
        for line in all_lines:
            if line not in unique_lines:
                unique_lines.append(line)
        
        if len(unique_lines) < 2:
            print(f"\n❌ [Tesseract] Chỉ tìm thấy {len(unique_lines)} dòng MRZ hợp lệ")
            return None
        
        line1, line2 = unique_lines[0], unique_lines[1]
        
        print(f"\n✅ [Tesseract] Tìm thấy MRZ:")
        print(f"   Line 1: {line1}")
        print(f"   Line 2: {line2}")
        
        # Parse MRZ thủ công
        result = parse_mrz_manual(line1, line2)
        result["method"] = "Tesseract OCR"
        
        return result
        
    except Exception as e:
        print(f"❌ [Tesseract] Lỗi: {e}")
        return None

def parse_mrz_manual(line1, line2):
    """Parse MRZ thủ công - chỉ 7 trường"""
    issuing = line1[2:5]
    name_part = line1[5:44]
    parts = name_part.split('<<')
    surname = parts[0].replace('<', ' ').strip() if parts else ''
    given = parts[1].replace('<', ' ').strip() if len(parts) > 1 else ''
    
    # Làm sạch tên
    surname = clean_name(surname)
    given = clean_name(given)
    
    passport = line2[0:9].replace('<', '').strip()
    nationality = line2[10:13]
    dob_raw = line2[13:19]
    sex = line2[20]
    expiry_raw = line2[21:27]
    
    # Giới tính: M hoặc F
    gender = 'M' if sex == 'M' else 'F' if sex == 'F' else ''
    
    return {
        "full_name": f"{surname} {given}".strip(),
        "passport_number": passport,
        "dob": format_date(dob_raw),
        "gender": gender,
        "issuing_country": issuing,
        "nationality": nationality,
        "expiry_date": format_date(expiry_raw)
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python read_mrz.py <image_path.jpg>")
        print("\nVí dụ:")
        print("  python read_mrz.py passport.jpg")
        print("  python read_mrz.py C:\\photos\\passport.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    print(f"📸 Đang đọc ảnh: {image_path}")
    
    # Thử PassportEye trước
    result = read_with_passporteye(image_path)
    
    # Nếu fail, thử Tesseract
    if not result:
        result = read_with_tesseract(image_path)
    
    if not result:
        print("\n❌ Không thể đọc MRZ!")
        sys.exit(1)
    
    # In kết quả
    print("\n" + "="*60)
    print("📊 KẾT QUẢ")
    print("="*60)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Lưu JSON
    output_file = image_path.rsplit('.', 1)[0] + '_mrz.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Đã lưu: {output_file}")
    print("✅ Hoàn thành!")

if __name__ == "__main__":
    main()
