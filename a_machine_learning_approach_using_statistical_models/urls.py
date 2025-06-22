"""a_machine_learning_approach_using_statistical_models URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path, re_path
from django.contrib import admin
from Remote_User import views as remoteuser
from a_machine_learning_approach_using_statistical_models import settings
from Service_Provider import views as serviceprovider
from django.conf.urls.static import static


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', remoteuser.login, name="login"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('Predict_Cardiac_Arrest_Type/', remoteuser.Predict_Cardiac_Arrest_Type, name="Predict_Cardiac_Arrest_Type"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    re_path(r'^charts/(?P<chart_type>\w+)/$', serviceprovider.charts, name="charts"),
    re_path(r'^charts1/(?P<chart_type>\w+)/$', serviceprovider.charts1, name="charts1"),
    re_path(r'^likeschart/(?P<like_chart>\w+)/$', serviceprovider.likeschart, name="likeschart"),
    path('View_Prediction_Of_Cardiac_Arrest_Type_Ratio/', serviceprovider.View_Prediction_Of_Cardiac_Arrest_Type_Ratio, name="View_Prediction_Of_Cardiac_Arrest_Type_Ratio"),
    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('View_Prediction_Of_Cardiac_Arrest_Type/', serviceprovider.View_Prediction_Of_Cardiac_Arrest_Type, name="View_Prediction_Of_Cardiac_Arrest_Type"),
    path('Download_Predicted_DataSets/', serviceprovider.Download_Predicted_DataSets, name="Download_Predicted_DataSets"),
    path('View_User_Training_Comparison/', serviceprovider.View_User_Training_Comparison, name="View_User_Training_Comparison"),

]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
