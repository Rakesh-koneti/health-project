"""Multiple_Disease_Detection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from .import views as mainview
from admins import views as admins
from users import views as usr
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',mainview.base),
    path("index/", mainview.index, name="index"),
    path("AdminLogin/", mainview.AdminLogin, name="AdminLogin"),
    path("UserLogin/", mainview.UserLogin, name="UserLogin"),
    path("UserRegister/", mainview.UserRegister, name="UserRegister"),

    # adminviews
    path("AdminLoginCheck/", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivaUsers/', admins.ActivaUsers, name='ActivaUsers'),
    path('DeleteUser/', admins.DeleteUser, name='DeleteUser'),
    path('BlockUser/', admins.BlockUser, name='BlockUser'),
    path('BlockUser/', admins.BlockUser, name='BlockUser'),
    path('UnblockUser/', admins.UnblockUser, name='UnblockUser'),



    # User Views

    path("UserRegisterActions/", usr.UserRegisterActions,name="UserRegisterActions"),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome/", usr.UserHome, name="UserHome"),
    path("view_data/", usr.view_data, name="view_data"),
    path("training/", usr.training, name="training"),
    path("prediction/", usr.prediction, name="prediction"),
    
  


]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
