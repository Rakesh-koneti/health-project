from django.db import models
from django.utils import timezone

# Create your models here.


class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(unique=True, max_length=100)
    email = models.CharField(unique=True, max_length=100)
    locality = models.CharField(max_length=100)
    date_joined = models.DateTimeField(default=timezone.now)

    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'UserRegistrations'


from django.db import models
from users.models import UserRegistrationModel

class PredictionHistory(models.Model):
    user = models.ForeignKey(UserRegistrationModel, on_delete=models.CASCADE)
    predicted_disease = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.loginid} - {self.predicted_disease}"



    