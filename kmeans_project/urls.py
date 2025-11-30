from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect

urlpatterns = [
    path('admin/', admin.site.urls),

    # Tambahkan routing root (/) → cluster-input
    path('', lambda request: redirect('cluster_input')),

    path('', include('clustering.urls')),
]
