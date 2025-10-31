# 📖 Đọc MRZ từ ảnh Passport

Script Python đơn giản để đọc mã MRZ (Machine Readable Zone) từ ảnh passport và xuất thông tin dạng JSON.

---

## ✨ Tính năng

-   ✅ Đọc MRZ từ ảnh passport (JPG/PNG)
-   ✅ Sử dụng 2 phương pháp: **PassportEye** (nhanh) và **Tesseract OCR** (backup)
-   ✅ Tự động làm sạch tên (loại bỏ ký tự thừa)
-   ✅ Xuất 7 trường quan trọng: tên, passport, ngày sinh, giới tính, quốc gia cấp, quốc tịch, ngày hết hạn
-   ✅ Lưu kết quả ra file JSON
-   ✅ Hỗ trợ điền tự động vào Smile PMS (optional)

---

## 🚀 Cài đặt nhanh

### 1. Cài Python 3.10+

Tải từ: https://www.python.org/downloads/

### 2. Cài thư viện

```bash
cd "C:\Users\ADMIN\Desktop\huy auto guest"
pip install -r requirements.txt
```

### 3. (Optional) Cài Tesseract OCR

Chỉ cần nếu PassportEye không đọc được MRZ:

-   Tải: https://github.com/UB-Mannheim/tesseract/wiki
-   Thêm vào PATH

---

## 📝 Cách sử dụng

### Đơn giản nhất

```bash
python read_mrz.py passport.jpg
```

### Với đường dẫn đầy đủ

```bash
python read_mrz.py "C:\Photos\passport.jpg"
```

### Output

**Console:**

````
{
    "method": "PassportEye",
    "full_name": "NGUYEN THI TRANG",
    "passport_number": "P032692896",
    "dob": "18/01/1998",
    "gender": "F",
    "issuing_country": "VNM",
    "nationality": "VNM",
    "expiry_date": "12/08/2034"
}
```

**File output:** `passport_mrz.json`

---

## 🐛 Debug

Script sẽ tự động tạo các file debug:

-   `debug_tesseract_15.jpg`: Ảnh đã xử lý (crop 15%)
-   `debug_tesseract_20.jpg`: Ảnh đã xử lý (crop 20%)
-   `debug_tesseract_25.jpg`: Ảnh đã xử lý (crop 25%)

Kiểm tra các file này để xem ảnh MRZ có rõ nét không.

---

## 📤 Output

-   **Console**: In kết quả JSON ra màn hình
-   **File**: Lưu file `<tên_ảnh>_mrz.json`

---

## 📊 Ví dụ output

```json
{
    "method": "PassportEye",
    "full_name": "NGUYEN THI TRANG",
    "passport_number": "P032692896",
    "dob": "18/01/1998",
    "gender": "F",
    "issuing_country": "VNM",
    "nationality": "VNM",
    "expiry_date": "12/08/2034"
}
```
````


 python -m venv venv

 venv\Scripts\activate