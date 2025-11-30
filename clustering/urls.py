from django.urls import path
from . import views

urlpatterns = [
    path('cluster-input/', views.cluster_input, name='cluster_input'),
    path('proses-kmeans/', views.proses_kmeans, name='proses_kmeans'),
    path('elbow/', views.elbow, name='elbow'),
    path('clustering/', views.clustering, name='clustering'),
    path('kesimpulan/', views.kesimpulan, name='kesimpulan'),
]
