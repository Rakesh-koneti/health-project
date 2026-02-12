from django.shortcuts import render,redirect
from users.models import UserRegistrationModel
from django.contrib import messages
# Create your views here.


def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return redirect('AdminHome')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

from django.shortcuts import render
from users.models import UserRegistrationModel
from django.utils.timesince import timesince
from datetime import timezone, datetime

def AdminHome(request):
    total_users = UserRegistrationModel.objects.count()
    activated_users = UserRegistrationModel.objects.filter(status='activated').count()
    blocked_users = UserRegistrationModel.objects.filter(status='blocked').count()
    waiting_users = UserRegistrationModel.objects.filter(status='waiting').count()

    recent_users = UserRegistrationModel.objects.order_by('-id')[:5]
    recent_activity = []
    for user in recent_users:
        recent_activity.append({
            'email': user.email,
            'time': timesince(user.date_joined) + " ago" if hasattr(user, 'date_joined') else "recently",
        })

    context = {
        'total_users': total_users,
        'activated_users': activated_users,
        'blocked_users': blocked_users,
        'waiting_users': waiting_users,
        'recent_activity': recent_activity,
    }

    return render(request, 'admins/AdminHome.html', context)



def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewusers.html', {'data': data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})

def DeleteUser(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        print("Deleting user with ID =", id)
        UserRegistrationModel.objects.filter(id=id).delete()
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})

def BlockUser(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'blocked'
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})



def BlockUser(request):
    if request.method == 'GET':
        uid = request.GET.get('uid')
        if uid:
            UserRegistrationModel.objects.filter(id=uid).update(status='blocked')
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})

def UnblockUser(request):
    if request.method == 'GET':
        uid = request.GET.get('uid')
        if uid:
            UserRegistrationModel.objects.filter(id=uid).update(status='activated')
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


# Create your views here.

