from pywinauto import Application
import json
import time

def load_config():
    """Load config.json"""
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def fill_smile(data, config=None):
    """Điền dữ liệu vào Smile PMS"""
    if config is None:
        config = load_config()
    
    try:
        # Kết nối với Smile PMS
        print("🔍 Đang tìm cửa sổ Smile PMS...")
        app = Application(backend="uia").connect(title_re=config["pms_window_title"], timeout=10)
        win = app.window(title_re=config["pms_window_title"])
        
        print("✅ Đã kết nối với Smile PMS")
        
        # Mapping từ config
        field_map = config["field_mapping"]
        
        # 1. Điền Last Name (full name)
        try:
            full_name = data.get('full_name', '')
            if full_name:
                print(f"✏️ Điền Last Name: {full_name}")
                ctrl = win.child_window(auto_id=field_map.get("last_name"), control_type="Edit")
                ctrl.set_focus()
                time.sleep(0.2)
                ctrl.set_text(full_name)
        except Exception as e:
            print(f"⚠️ Không thể điền Last Name: {e}")
        
        # 2. Tick Gender (Male/Female checkbox)
        try:
            gender = data.get('gender', '')
            if gender:
                print(f"✏️ Chọn Gender: {gender}")
                if gender == "M":
                    # Tick Male checkbox
                    ctrl = win.child_window(auto_id=field_map.get("gender_male"), control_type="CheckBox")
                    ctrl.set_focus()
                    time.sleep(0.2)
                    if ctrl.get_toggle_state() != 1:
                        ctrl.toggle()
                elif gender == "F":
                    # Female là mặc định (không tick Male)
                    ctrl = win.child_window(auto_id=field_map.get("gender_male"), control_type="CheckBox")
                    ctrl.set_focus()
                    time.sleep(0.2)
                    if ctrl.get_toggle_state() == 1:
                        ctrl.toggle()
        except Exception as e:
            print(f"⚠️ Không thể chọn Gender: {e}")
        
        # 3. Điền Date of Birth
        try:
            dob = data.get('dob', '')
            if dob:
                print(f"✏️ Điền Date of Birth: {dob}")
                ctrl = win.child_window(auto_id=field_map.get("dob"), control_type="Edit")
                ctrl.set_focus()
                time.sleep(0.2)
                ctrl.set_text(dob)
        except Exception as e:
            print(f"⚠️ Không thể điền Date of Birth: {e}")
        
        # 4. Điền Country (issuing_country)
        try:
            country = data.get('issuing_country', '')
            if country:
                print(f"✏️ Điền Country: {country}")
                ctrl = win.child_window(auto_id=field_map.get("country"), control_type="Edit")
                ctrl.set_focus()
                time.sleep(0.2)
                ctrl.set_text(country)
        except Exception as e:
            print(f"⚠️ Không thể điền Country: {e}")
        
        # 5. Điền Nationality
        try:
            nationality = data.get('nationality', '')
            if nationality:
                print(f"✏️ Điền Nationality: {nationality}")
                ctrl = win.child_window(auto_id=field_map.get("nationality"), control_type="Edit")
                ctrl.set_focus()
                time.sleep(0.2)
                ctrl.set_text(nationality)
        except Exception as e:
            print(f"⚠️ Không thể điền Nationality: {e}")
        
        # 6. Điền Passport/ID
        try:
            passport = data.get('passport_number', '')
            if passport:
                print(f"✏️ Điền Passport/ID: {passport}")
                ctrl = win.child_window(auto_id=field_map.get("passport"), control_type="Edit")
                ctrl.set_focus()
                time.sleep(0.2)
                ctrl.set_text(passport)
        except Exception as e:
            print(f"⚠️ Không thể điền Passport/ID: {e}")
        
        print("✅ Đã điền xong dữ liệu")
        print("⚠️ Vui lòng kiểm tra và xác nhận trong Smile PMS")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi kết nối Smile PMS: {e}")
        print("💡 Kiểm tra:")
        print("   - Smile PMS đã mở chưa?")
        print("   - Tiêu đề cửa sổ trong config.json đúng chưa?")
        print("   - Chạy Python với quyền admin")
        return False

def fill_from_json(json_path):
    """Đọc JSON và điền vào Smile PMS"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return fill_smile(data)
