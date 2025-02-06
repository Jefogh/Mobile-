[app]
# اسم التطبيق وعنوان الحزمة
title = MyCaptchaApp
package.name = mycaptchaapp
package.domain = org.example

# الملفات التي يجب تضمينها
source.include_exts = py,png,jpg,kv,pth

# المتطلبات: تأكد من تضمين جميع المكتبات التي يستخدمها التطبيق
requirements = python3,kivy,opencv-python,torch,torchvision,Pillow,requests,numpy

# اتجاه الشاشة (على سبيل المثال portrait)
orientation = portrait

# إذا كنت تستخدم ملفات إضافية أو مجلد assets، يمكنك إضافتها:
source.include_patterns = assets/*, squeezenet_trained.pth

# إعدادات أخرى متعلقة بالنظام
version = 0.1
android.permissions = INTERNET

[buildozer]
# إعدادات Buildozer (يمكن تعديلها حسب الحاجة)
log_level = 2
