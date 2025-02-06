import base64
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import requests

# ================================
# الجزء الخاص بنموذج الذكاء الاصطناعي
# ================================

# استخدام جهاز CPU (يمكن تعديل ذلك لدعم GPU عند توفره)
cpu_device = torch.device("cpu")


class TrainedModel:
    def __init__(self):
        start_time = time.time()
        # استخدام SqueezeNet 1_1 بدلاً من SqueezeNet 1_0
        self.model = models.squeezenet1_1(pretrained=False)
        # تعديل الطبقة النهائية لتخرج 30 قيمة كما تم أثناء التدريب
        self.model.classifier[1] = nn.Conv2d(512, 30, kernel_size=1)
        model_path = "squeezenet_trained.pth"  # تأكد من أن الملف في نفس المجلد أو ضبط المسار
        self.model.load_state_dict(torch.load(model_path, map_location=cpu_device))
        self.model = self.model.to(cpu_device)
        self.model.eval()
        print(f"Model loaded in {time.time() - start_time:.4f} seconds")

    def predict(self, img):
        start_time = time.time()
        # تغيير حجم الصورة لتتوافق مع مدخلات النموذج
        resized_image = cv2.resize(img, (224, 224))
        print(f"Image resizing (OpenCV) took {time.time() - start_time:.4f} seconds")

        # تحويل الصورة إلى PIL مع تصحيح ترتيب القنوات (BGR -> RGB)
        pil_image = Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        # إعداد ما قبل المعالجة بنفس ما استخدم أثناء التدريب
        preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        tensor_image = preprocess(pil_image).unsqueeze(0).to(cpu_device)
        print(f"Image preprocessing took {time.time() - start_time:.4f} seconds")

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(tensor_image).view(-1, 30)
        print(f"Model prediction took {time.time() - start_time:.4f} seconds")

        # تقسيم المخرجات إلى ثلاث مجموعات
        num1_preds = outputs[:, :10]
        operation_preds = outputs[:, 10:13]
        num2_preds = outputs[:, 13:]

        _, num1_predicted = torch.max(num1_preds, 1)
        _, operation_predicted = torch.max(operation_preds, 1)
        _, num2_predicted = torch.max(num2_preds, 1)

        operation_map = {0: "+", 1: "-", 2: "×"}
        predicted_operation = operation_map.get(operation_predicted.item(), "?")

        del tensor_image
        return num1_predicted.item(), predicted_operation, num2_predicted.item()


# ================================
# جزء الواجهة: مؤثر الدوران المتوسع (ExpandingCircle)
# ================================
# هنا نقوم بإنشاء Widget مخصص لرسم دائرة متحركة على Canvas
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse


class ExpandingCircle(Widget):
    def __init__(self, x, y, max_radius, color=(0, 0, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.center_x = x
        self.center_y = y
        self.max_radius = max_radius
        self.radius = 10
        self.growing = True
        self.color = color
        self._update_event = None
        with self.canvas:
            Color(*self.color)
            self.ellipse = Ellipse(pos=(self.center_x - self.radius, self.center_y - self.radius),
                                   size=(self.radius * 2, self.radius * 2))
        self.start_animation()

    def start_animation(self):
        from kivy.clock import Clock
        self._update_event = Clock.schedule_interval(self.expand_circle, 0.05)

    def expand_circle(self, dt):
        if self.growing:
            self.radius += 2
            if self.radius >= self.max_radius:
                self.growing = False
        else:
            self.radius -= 2
            if self.radius <= 10:
                self.growing = True

        self.ellipse.pos = (self.center_x - self.radius, self.center_y - self.radius)
        self.ellipse.size = (self.radius * 2, self.radius * 2)

    def stop(self):
        if self._update_event:
            self._update_event.cancel()
            self.canvas.clear()


# ================================
# الطبقة الخلفية (Backend) للتعامل مع الحسابات والـ API
# ================================

class CaptchaAppBackend:
    def __init__(self):
        self.accounts = {}  # تخزين بيانات الحسابات
        self.background_images = []  # قائمة الخلفيات المحملة
        self.trained_model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.load_model()

    def load_model(self):
        print("Loading model...")
        start_time = time.time()
        self.trained_model = TrainedModel()
        print(f"Model loaded and ready in {time.time() - start_time:.4f} seconds")

    def generate_user_agent(self):
        user_agent_list = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_0) AppleWebKit/537.36 Chrome/100.0.4896.127 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 Version/15.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 12; SM-G998B) AppleWebKit/537.36 Chrome/102.0.5005.61 Mobile Safari/537.36",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:98.0) Gecko/20100101 Firefox/98.0",
            # المزيد من user agents حسب الحاجة
        ]
        return random.choice(user_agent_list)

    def create_session(self, user_agent):
        headers = {
            "User-Agent": user_agent,
            "host": "api.ecsc.gov.sy:8443",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "ar,en-US;q=0.7,en;q=0.3",
            "Referer": "https://ecsc.gov.sy/login",
            "Content-Type": "application/json",
            "Source": "WEB",
            "Origin": "https://ecsc.gov.sy",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "Priority": "u=1",
        }
        session = requests.Session()
        session.headers.update(headers)
        return session

    def login(self, username, password, session, retry_count=3):
        login_url = "https://api.ecsc.gov.sy:8443/secure/auth/login"
        login_data = {"username": username, "password": password}
        for attempt in range(retry_count):
            try:
                post_response = session.post(login_url, json=login_data, verify=False)
                if post_response.status_code == 200:
                    self.update_notification("Login successful.", "green", post_response.text)
                    return True
                else:
                    self.update_notification(f"Login failed. Status code: {post_response.status_code}",
                                             "red", post_response.text)
                    return False
            except requests.RequestException as e:
                self.update_notification(f"Request error: {e}", "red")
                return False
        return False

    def is_session_valid(self, session):
        try:
            test_url = "https://api.ecsc.gov.sy:8443/some_endpoint_to_check_session"
            response = session.get(test_url, verify=False)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def fetch_process_ids(self, session):
        try:
            url = "https://api.ecsc.gov.sy:8443/dbm/db/execute"
            payload = {
                "ALIAS": "OPkUVkYsyq",
                "P_USERNAME": "WebSite",
                "P_PAGE_INDEX": 0,
                "P_PAGE_SIZE": 100
            }
            headers = {
                "Content-Type": "application/json",
                "Alias": "OPkUVkYsyq",
                "Referer": "https://ecsc.gov.sy/requests",
                "Origin": "https://ecsc.gov.sy",
            }
            response = session.post(url, json=payload, headers=headers, verify=False)
            if response.status_code == 200:
                data = response.json()
                process_ids = data.get("P_RESULT", [])
                if process_ids:
                    return process_ids
                else:
                    self.update_notification("No process IDs found.", "red")
            else:
                self.update_notification(f"Failed to fetch process IDs. Status code: {response.status_code}", "red")
        except Exception as e:
            self.update_notification(f"Error fetching process IDs: {str(e)}", "red")
        return None

    def get_captcha(self, session, captcha_id, username):
        try:
            captcha_url = f"https://api.ecsc.gov.sy:8443/files/fs/captcha/{captcha_id}"
            while True:
                response = session.get(captcha_url, verify=False)
                self.update_notification(f"Server Response: {response.text}",
                                         "green" if response.status_code == 200 else "red")
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data.get("file")
                elif response.status_code == 429:
                    time.sleep(0.1)
                elif response.status_code in {401, 403}:
                    if self.login(username, self.accounts[username]["password"], session):
                        continue
                else:
                    break
        except Exception as e:
            self.update_notification(f"Error: {str(e)}", "red")
        return None

    def process_captcha(self, captcha_image):
        if not self.background_images:
            return captcha_image
        best_background = None
        min_diff = float("inf")
        for background in self.background_images:
            background = cv2.resize(background, (captcha_image.shape[1], captcha_image.shape[0]))
            processed_image = self.remove_background_keep_original_colors(captcha_image, background)
            gray_diff = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            score = np.sum(gray_diff)
            if score < min_diff:
                min_diff = score
                best_background = background
        if best_background is not None:
            cleaned_image = self.remove_background_keep_original_colors(captcha_image, best_background)
            return cleaned_image
        else:
            return captcha_image

    def remove_background_keep_original_colors(self, captcha_image, background_image):
        # تقليل الدقة لتسريع العملية
        scale_factor = 0.5
        captcha_resized = cv2.resize(captcha_image, (0, 0), fx=scale_factor, fy=scale_factor)
        background_resized = cv2.resize(background_image, (0, 0), fx=scale_factor, fy=scale_factor)
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            captcha_image_gpu = cv2.cuda_GpuMat()
            background_image_gpu = cv2.cuda_GpuMat()
            captcha_image_gpu.upload(captcha_resized)
            background_image_gpu.upload(background_resized)
            diff_gpu = cv2.cuda.absdiff(captcha_image_gpu, background_image_gpu)
            gray_gpu = cv2.cuda.cvtColor(diff_gpu, cv2.COLOR_BGR2GRAY)
            gray = gray_gpu.download()
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            mask_gpu = cv2.cuda_GpuMat()
            mask_gpu.upload(mask)
            result_gpu = cv2.cuda.bitwise_and(captcha_image_gpu, captcha_image_gpu, mask=mask_gpu)
            result = result_gpu.download()
            return result
        else:
            diff = cv2.absdiff(captcha_resized, background_resized)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            result = cv2.bitwise_and(captcha_resized, captcha_resized, mask=mask)
            return result

    def solve_captcha_from_prediction(self, prediction):
        num1, operation, num2 = prediction
        if operation == "+":
            return num1 + num2
        elif operation == "-":
            return abs(num1 - num2)
        elif operation == "×":
            return num1 * num2
        return None

    # دالة مبدئية لتحديث الإشعارات في الواجهة (سيتم الربط معها من واجهة Kivy)
    def update_notification(self, message, color, response_text=None):
        full_message = message
        if response_text:
            full_message += f"\nServer Response: {response_text}"
        print(f"[{color.upper()}] {full_message}")


# ================================
# واجهة المستخدم باستخدام Kivy
# ================================
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.filechooser import FileChooserListView
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
import io


# نافذة إدخال بيانات (لإضافة حساب أو إدخال معرف الكابتشا)
class InputPopup(Popup):
    def __init__(self, title, hint_text="", password=False, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.size_hint = (0.8, 0.4)
        self.auto_dismiss = False
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        self.text_input = TextInput(hint_text=hint_text, multiline=False, password=password)
        layout.add_widget(self.text_input)
        btn_layout = BoxLayout(size_hint_y=0.3)
        ok_btn = Button(text="OK")
        cancel_btn = Button(text="Cancel")
        ok_btn.bind(on_release=self.dismiss_with_value)
        cancel_btn.bind(on_release=self.dismiss)
        btn_layout.add_widget(ok_btn)
        btn_layout.add_widget(cancel_btn)
        layout.add_widget(btn_layout)
        self.content = layout
        self.value = None

    def dismiss_with_value(self, instance):
        self.value = self.text_input.text
        self.dismiss()


# الواجهة الرئيسية للتطبيق
class MainScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # الاتجاه العمودي للمحتوى
        self.orientation = "vertical"
        # طبقة للتنبيهات
        self.notification_label = Label(text="Welcome!", size_hint_y=0.1, color=(1, 1, 1, 1), bold=True)
        self.add_widget(self.notification_label)
        # طبقة الأزرار الرئيسية
        btn_layout = BoxLayout(size_hint_y=0.1)
        add_account_btn = Button(text="Add Account")
        upload_bg_btn = Button(text="Upload Backgrounds")
        login_btn = Button(text="Login Saved Accounts")
        add_account_btn.bind(on_release=self.add_account)
        upload_bg_btn.bind(on_release=self.upload_backgrounds)
        login_btn.bind(on_release=self.login_saved_accounts)
        btn_layout.add_widget(add_account_btn)
        btn_layout.add_widget(upload_bg_btn)
        btn_layout.add_widget(login_btn)
        self.add_widget(btn_layout)
        # منطقة عرض محتوى متنوع (مثل الحسابات والكابتشا)
        self.content_scroll = ScrollView()
        self.content_layout = BoxLayout(orientation="vertical", size_hint_y=None)
        self.content_layout.bind(minimum_height=self.content_layout.setter('height'))
        self.content_scroll.add_widget(self.content_layout)
        self.add_widget(self.content_scroll)
        # منطقة عرض صورة الكابتشا
        self.captcha_image = KivyImage(size_hint_y=0.5)
        self.add_widget(self.captcha_image)
        # منطقة عرض الوقت/تفاصيل الأداء
        self.time_label = Label(text="", size_hint_y=0.1)
        self.add_widget(self.time_label)

        # إنشاء نسخة من الـ backend
        self.backend = CaptchaAppBackend()

    def show_notification(self, message, color="blue"):
        self.notification_label.text = message
        # يمكن إضافة تأثيرات زمنية لإعادة التعيين
        Clock.schedule_once(lambda dt: self.clear_notification(), 8)

    def clear_notification(self):
        self.notification_label.text = ""

    def add_account(self, instance):
        # استخدام سلسلة من النوافذ المنبثقة لجمع بيانات الحساب
        content = BoxLayout(orientation="vertical", spacing=10)
        username_popup = InputPopup(title="Enter Username", hint_text="Username")
        username_popup.bind(on_dismiss=lambda instance: self._got_username(username_popup.value))
        username_popup.open()

    def _got_username(self, username):
        if not username:
            self.show_notification("Username cannot be empty", "red")
            return
        pwd_popup = InputPopup(title="Enter Password", hint_text="Password", password=True)
        pwd_popup.bind(on_dismiss=lambda instance: self._got_password(username, pwd_popup.value))
        pwd_popup.open()

    def _got_password(self, username, password):
        if not password:
            self.show_notification("Password cannot be empty", "red")
            return
        # إنشاء جلسة وتسجيل الدخول
        user_agent = self.backend.generate_user_agent()
        session = self.backend.create_session(user_agent)
        start_time = time.time()

        def login_thread():
            if self.backend.login(username, password, session):
                elapsed_time = time.time() - start_time
                self.backend.accounts[username] = {
                    "password": password,
                    "user_agent": user_agent,
                    "session": session,
                    "captcha_id1": None,
                    "captcha_id2": None,
                }
                self.show_notification(f"Login successful for user {username}. Time: {elapsed_time:.2f}s", "green")
                # بعد تسجيل الدخول يمكن جلب بيانات العمليات (process IDs)
                process_data = self.backend.fetch_process_ids(session)
                if process_data:
                    Clock.schedule_once(lambda dt: self.create_account_ui(username, process_data))
                else:
                    self.show_notification(f"Failed to fetch process IDs for user {username}.", "red")
            else:
                elapsed_time = time.time() - start_time
                self.show_notification(f"Failed to login for user {username}. Time: {elapsed_time:.2f}s", "red")

        threading.Thread(target=login_thread).start()

    def create_account_ui(self, username, process_data):
        account_box = BoxLayout(orientation="vertical", size_hint_y=None, height=200, padding=5, spacing=5)
        account_box.add_widget(Label(text=f"Account: {username}", size_hint_y=0.2))
        for process in process_data:
            process_id = process.get("PROCESS_ID")
            center_name = process.get("ZCENTER_NAME", "Unknown Center")
            process_box = BoxLayout(size_hint_y=0.2, spacing=5)
            proc_btn = Button(text=center_name, font_size=14)
            proc_btn.bind(on_release=lambda inst, pid=process_id: self.request_captcha(username, pid))
            process_box.add_widget(proc_btn)
            account_box.add_widget(process_box)
        self.content_layout.add_widget(account_box)

    def login_saved_accounts(self, instance):
        for username, account_info in self.backend.accounts.items():
            session = account_info.get("session")
            if not session or not self.backend.is_session_valid(session):
                user_agent = self.backend.generate_user_agent()
                session = self.backend.create_session(user_agent)
                password = account_info.get("password")
                if self.backend.login(username, password, session):
                    self.backend.accounts[username]["session"] = session
                    self.show_notification(f"Login successful for {username}", "green")
                else:
                    self.show_notification(f"Login failed for {username}", "red")

    def request_captcha(self, username, captcha_id):
        # عرض مؤشر (spinner) باستخدام ExpandingCircle فوق Widget مؤقتاً
        spinner = ExpandingCircle(x=self.width/2, y=self.height/2, max_radius=30, color=(0, 0, 1, 1))
        self.add_widget(spinner)

        def request_thread():
            session = self.backend.accounts[username].get("session")
            if not session:
                self.show_notification(f"No session found for user {username}", "red")
                Clock.schedule_once(lambda dt: self.remove_widget(spinner))
                return

            captcha_data = self.backend.get_captcha(session, captcha_id, username)
            if captcha_data:
                # عرض صورة الكابتشا والقيام بعمليات المعالجة
                self.executor_submit(self.show_captcha, captcha_data, username, captcha_id)
            Clock.schedule_once(lambda dt: self.remove_widget(spinner), 0)

        threading.Thread(target=request_thread).start()

    def executor_submit(self, func, *args, **kwargs):
        # استخدام ThreadPoolExecutor من الـ backend لتشغيل الوظائف بدون تجميد الواجهة
        self.backend.executor.submit(func, *args, **kwargs)

    def show_captcha(self, captcha_data, username, captcha_id):
        try:
            # فك تشفير الصورة من base64
            if "," in captcha_data:
                captcha_base64 = captcha_data.split(",")[1]
            else:
                captcha_base64 = captcha_data
            captcha_image_data = np.frombuffer(base64.b64decode(captcha_base64), dtype=np.uint8)
            captcha_image_cv = cv2.imdecode(captcha_image_data, cv2.IMREAD_COLOR)
            if captcha_image_cv is None:
                print("Failed to decode captcha image from memory.")
                return
            start_time = time.time()
            processed_image = self.backend.process_captcha(captcha_image_cv)
            processed_image = cv2.resize(processed_image, (200, 114))
            elapsed_time_bg_removal = time.time() - start_time

            # تحويل الصورة إلى texture للعرض في Kivy
            buf = cv2.flip(processed_image, 0).tobytes()
            image_texture = CoreImage(io.BytesIO(buf), ext="png").texture

            def update_image(dt):
                self.captcha_image.texture = image_texture

            Clock.schedule_once(update_image, 0)

            start_time = time.time()
            predictions = self.backend.trained_model.predict(processed_image)
            elapsed_time_prediction = time.time() - start_time
            ocr_output_text = f"{predictions[0]} {predictions[1]} {predictions[2]}"
            print(f"Predicted Operation: {ocr_output_text}")
            self.show_notification(f"Captcha solved in {elapsed_time_prediction:.2f}s", "green")
            self.time_label.text = f"Background removal: {elapsed_time_bg_removal:.2f}s, Prediction: {elapsed_time_prediction:.2f}s"
            captcha_solution = self.backend.solve_captcha_from_prediction(predictions)
            if captcha_solution is not None:
                self.executor_submit(self.submit_captcha, username, captcha_id, captcha_solution)
        except Exception as e:
            self.show_notification(f"Failed to show captcha: {e}", "red")

    def submit_captcha(self, username, captcha_id, captcha_solution):
        session = self.backend.accounts[username].get("session")
        if not session:
            self.show_notification(f"No session found for user {username}", "red")
            return
        try:
            get_url = f"https://api.ecsc.gov.sy:8443/rs/reserve?id={captcha_id}&captcha={captcha_solution}"
            response = session.get(get_url, verify=False)
            self.show_notification(f"Server Response: {response.text}",
                                   "green" if response.status_code == 200 else "red")
        except Exception as e:
            self.show_notification(f"Failed to submit captcha: {e}", "red")

    def upload_backgrounds(self, instance):
        # عرض FileChooser داخل Popup لتحميل صور الخلفية
        filechooser = FileChooserListView(filters=["*.jpg", "*.png", "*.jpeg"])
        popup = Popup(title="Select Background Images", content=filechooser, size_hint=(0.9, 0.9))

        def on_selection(instance, selection):
            if selection:
                self.backend.background_images = []
                for path in selection:
                    bg = cv2.imread(path)
                    if bg is not None:
                        self.backend.background_images.append(bg)
                self.show_notification(f"{len(selection)} background images uploaded and preprocessed successfully!", "green")
                popup.dismiss()

        filechooser.bind(on_submit=on_selection)
        popup.open()


# ================================
# تطبيق Kivy الرئيسي
# ================================
class CaptchaKivyApp(App):
    def build(self):
        return MainScreen()


if __name__ == "__main__":
    CaptchaKivyApp().run()
