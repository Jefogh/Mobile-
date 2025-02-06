[app]
# عنوان التطبيق وبيانات الحزمة
title = MyCaptchaApp
package.name = mycaptchaapp
package.domain = org.example

# تحديد الدليل الذي يحتوي على ملفات المصدر (في هذه الحالة الدليل الحالي)
source.dir = .

# الملفات التي يجب تضمينها (يمكنك تعديل القائمة بحسب احتياجات المشروع)
source.include_exts = py,png,jpg,kv,pth

# المتطلبات (تأكد من تضمين جميع المكتبات المستخدمة)
requirements = python3,kivy,opencv-python,torch,torchvision,Pillow,requests,numpy

# اتجاه الشاشة (مثلاً portrait)
orientation = portrait

# تضمين ملفات إضافية (مثل ملفات الخلفيات أو النموذج)
source.include_patterns = assets/*, squeezenet_trained.pth

# إعدادات أخرى
version = 0.1
android.permissions = INTERNET

[buildozer]
log_level = 2
