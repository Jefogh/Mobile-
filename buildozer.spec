[app]
# معلومات التطبيق الأساسية
title = MyCaptchaApp
package.name = mycaptchaapp
package.domain = org.example

# تحديد مجلد ملفات المصدر (حاليًا الدليل الحالي)
source.dir = .
# الملفات التي يجب تضمينها في البنية
source.include_exts = py,png,jpg,kv,pth
# تضمين ملفات إضافية (مثل الملفات داخل مجلد assets وملف النموذج)
source.include_patterns = assets/*, squeezenet_trained.pth

# رقم إصدار التطبيق
version = 0.1

# المتطلبات (تأكد من تضمين جميع المكتبات التي يستخدمها التطبيق)
requirements = python3,kivy,opencv-python,torch,torchvision,Pillow,requests,numpy

# إعدادات شاشة التطبيق
orientation = portrait

# أذونات Android المطلوبة
android.permissions = INTERNET

# إعدادات SDK و Build Tools (نتجنب بذلك إصدارات المعاينة)
android.sdk = 30
android.sdk_build_tools_version = 20.0.1
android.minapi = 21
# تحديد إصدار NDK (يمكنك تعديله حسب احتياجاتك)
android.ndk = 22b

# إعدادات التخزين الخاص (اختياري)
android.private_storage = True

[buildozer]
# مستوى السجل (log level) للتحكم بالمخرجات
log_level = 2
# تحذير عند تشغيل buildozer كـ root (يُفضل عدم تشغيله كـ root)
warn_on_root = 1
