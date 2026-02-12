from django import forms
from .models import UserRegistrationModel

class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[a-zA-Z ]+',
            'placeholder': 'Enter your full name'
        }),
        required=True,
        max_length=100
    )

    loginid = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[a-zA-Z0-9]+',
            'placeholder': 'Choose a unique login ID'
        }),
        required=True,
        max_length=100
    )

    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'pattern': '(?=.*\\d)(?=.*[a-z])(?=.*[A-Z]).{8,}',
            'title': 'Must contain at least one number, one uppercase and lowercase letter, and be at least 8 characters',
            'placeholder': 'Create a strong password'
        }),
        required=True,
        max_length=100
    )

    mobile = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'pattern': '[6-9][0-9]{9}',
            'placeholder': 'Enter your 10-digit mobile number'
        }),
        required=True,
        max_length=100
    )

    email = forms.CharField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address'
        }),
        required=True,
        max_length=100
    )

    locality = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your locality'
        }),
        required=True,
        max_length=100
    )

    status = forms.CharField(
        widget=forms.HiddenInput(),
        initial='waiting',
        max_length=100
    )

    class Meta:
        model = UserRegistrationModel
        fields = '__all__'

    # ✅ Field-level uniqueness checks
    def clean_email(self):
        email = self.cleaned_data.get('email')
        if UserRegistrationModel.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("Email already exists.")
        return email

    def clean_mobile(self):
        mobile = self.cleaned_data.get('mobile')
        if UserRegistrationModel.objects.filter(mobile=mobile).exists():
            raise forms.ValidationError("Mobile number already exists.")
        return mobile

    def clean_loginid(self):
        loginid = self.cleaned_data.get('loginid')
        if UserRegistrationModel.objects.filter(loginid__iexact=loginid).exists():
            raise forms.ValidationError("Login ID already exists.")
        return loginid
