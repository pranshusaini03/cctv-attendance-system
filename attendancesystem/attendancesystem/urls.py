from django.contrib import admin
from django.urls import path
from attendancesystem import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main, name='home'),  # Main page
    path('attendance/', views.attendance, name='attendance'),
    path('register/', views.register, name='register'),
    path('view/', views.view_attendance, name='view_attendance'),
    path('startattendancednn/<int:device>/', views.startattendancednn, name='startattendancednn'),
    path('startattendancemtcnn/<int:device>/', views.startattendancemtcnn, name='startattendancemtcnn'),
    path('startattendanceht/<int:device>/', views.startattendanceht, name='startattendanceht'),
    path('new_student/', views.new_student, name='new_student'),
    path('train_model/', views.model ,name='model')
]
