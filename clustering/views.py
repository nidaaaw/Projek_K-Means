from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

# -------------------------------------------------
# FUNGSI KMEANS MANUAL
# -------------------------------------------------
def kmeans(X, k, max_iters=300, random_state=0):
    np.random.seed(random_state)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    wcss = np.sum((X - centroids[labels]) ** 2)
    return labels, centroids, wcss


# =================================================
# 1️⃣ INPUT JUMLAH K
# =================================================
def cluster_input(request):
    if request.method == "POST":
        request.session["k"] = int(request.POST.get("k"))
        return redirect("proses_kmeans")

    return render(request, 'clustering/menentukan_cluster.html')


# =================================================
# 2️⃣ PROSES KMEANS
# =================================================
def proses_kmeans(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'clustering', 'Mall_Customers.csv'))

    k = int(request.session.get("k", 3))

    # Standardisasi kolom
    df = df.rename(columns={
        "Age": "age",
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending"
    })

    X = df[['age', 'income', 'spending']].values

    # Centroid awal manual
    np.random.seed(0)
    initial_centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    # Hitung jarak
    distances = np.linalg.norm(X[:, None] - initial_centroids[None, :], axis=2)
    labels = np.argmin(distances, axis=1)

    distance_df = pd.DataFrame(distances, columns=[f"C{i}" for i in range(k)])
    centroid_awal = pd.DataFrame(initial_centroids, columns=['age', 'income', 'spending'])
    centroid_akhir = pd.DataFrame(
        [X[labels == i].mean(axis=0) for i in range(k)],
        columns=['age', 'income', 'spending']
    )

    df["cluster"] = labels

    context = {
        "k": k,
        "data_asli": df.head(200).to_html(classes="table table-bordered"),
        "distance_table": distance_df.to_html(classes="table table-bordered table-sm"),
        "centroid_awal": centroid_awal.to_html(classes="table table-bordered"),
        "centroid_akhir": centroid_akhir.to_html(classes="table table-bordered"),
        "cluster_hasil": df[['age', 'income', 'spending', 'cluster']].head(200)
                        .to_html(classes="table table-bordered"),
    }
    return render(request, "clustering/proses_kmeans.html", context)


# =================================================
# 3️⃣ HALAMAN CLUSTERING (HASIL AKHIR)
# =================================================
def clustering(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'clustering', 'Mall_Customers.csv'))

    # Standardisasi kolom
    df = df.rename(columns={
        "Age": "age",
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending"
    })

    X = df[['age', 'income', 'spending']].values

    k = int(request.session.get("k", 3))

    labels, centroids, wcss = kmeans(X, k)
    df["cluster"] = labels

    # ---------------------------------------------
    # BENTUKKAN dictionary DATA PER CLUSTER
    # ---------------------------------------------
    cluster_groups = {}
    for i in range(k):
        data_cluster_i = df[df["cluster"] == i][["age", "income", "spending", "cluster"]]


        # Samakan nama kolom dengan yang dipakai template
        data_cluster_i = data_cluster_i.rename(columns={
            "income": "Pendapatan",
            "spending": "Pengeluaran"
        })

        cluster_groups[i] = data_cluster_i.to_dict("records")

        

    # ---------------------------------------------
    # PLOT CLUSTER
    # ---------------------------------------------
    fig, ax = plt.subplots()
    for i in range(k):
        ax.scatter(X[labels == i, 1], X[labels == i, 2], label=f"Cluster {i}")

    ax.scatter(centroids[:, 1], centroids[:, 2], s=200, c='black', marker='X')
    ax.set_title("Visualisasi Cluster")
    ax.set_xlabel("Pendapatan (Income)")
    ax.set_ylabel("Pengeluaran (Spending)")
    ax.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    cluster_plot = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    # Summary cluster
    summary = df.groupby("cluster")[['age', 'income', 'spending']].mean().astype(int)

    return render(request, 'clustering/clustering.html', {
        'cluster_plot': cluster_plot,
        'summary': summary.to_html(classes="table table-bordered"),
        'cluster_groups': cluster_groups,
        'k': k,
    })


# =================================================
# 4️⃣ KESIMPULAN LENGKAP DENGAN SEMUA VISUALISASI
# =================================================
def kesimpulan(request):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, 'clustering', 'Mall_Customers.csv'))

    df = df.rename(columns={
        "Age": "age",
        "Annual Income (k$)": "income",
        "Spending Score (1-100)": "spending"
    })

    X = df[['age', 'income', 'spending']].values
    k = int(request.session.get("k", 3))

    # Jalankan clustering
    labels, centroids, wcss = kmeans(X, k)
    df["cluster"] = labels

    # ===========================================================
    # 1. GRAFIK SEBARAN DATA AWAL (BEFORE CLUSTERING)
    # ===========================================================
    fig1, ax1 = plt.subplots()
    ax1.scatter(df["income"], df["spending"])
    ax1.set_title("Sebaran Data Awal (Before Clustering)")
    ax1.set_xlabel("Pendapatan (Income)")
    ax1.set_ylabel("Pengeluaran (Spending)")

    buffer1 = BytesIO()
    plt.savefig(buffer1, format="png")
    plot_awal = base64.b64encode(buffer1.getvalue()).decode()
    plt.close()

    # ===========================================================
    # 2. GRAFIK ELBOW METHOD UNTUK MENENTUKAN K OPTIMAL
    # ===========================================================
    wcss_list = []
    for i in range(1, 10):
        _, _, wcss_i = kmeans(X, i)
        wcss_list.append(wcss_i)

    fig2, ax2 = plt.subplots()
    ax2.plot(range(1, 10), wcss_list, marker='o')
    ax2.set_title("Elbow Method (WCSS vs K)")
    ax2.set_xlabel("Jumlah Cluster (K)")
    ax2.set_ylabel("WCSS")

    buffer2 = BytesIO()
    plt.savefig(buffer2, format="png")
    elbow_plot = base64.b64encode(buffer2.getvalue()).decode()
    plt.close()

    # ===========================================================
    # 3. VISUALISASI CLUSTER AKHIR
    # ===========================================================
    fig3, ax3 = plt.subplots()
    for i in range(k):
        ax3.scatter(X[labels == i, 1], X[labels == i, 2], label=f"Cluster {i}")

    ax3.scatter(centroids[:, 1], centroids[:, 2], s=200, c="black", marker="X")
    ax3.set_title("Visualisasi Cluster Akhir")
    ax3.set_xlabel("Pendapatan (Income)")
    ax3.set_ylabel("Pengeluaran (Spending)")
    ax3.legend()

    buffer3 = BytesIO()
    plt.savefig(buffer3, format="png")
    cluster_plot = base64.b64encode(buffer3.getvalue()).decode()
    plt.close()

    # ===========================================================
    # 4. GRAPH BAR RATA-RATA CLUSTER
    # ===========================================================
    summary = df.groupby("cluster")[['age', 'income', 'spending']].mean()

    fig4, ax4 = plt.subplots()
    summary.plot(kind='bar', ax=ax4)
    ax4.set_title("Rata-Rata Tiap Cluster")
    ax4.set_xlabel("Cluster")
    ax4.set_ylabel("Nilai Rata-Rata")

    buffer4 = BytesIO()
    plt.savefig(buffer4, format="png")
    summary_plot = base64.b64encode(buffer4.getvalue()).decode()
    plt.close()

    # ===========================================================
    # KESIMPULAN OTOMATIS
    # ===========================================================
    kesimpulan_text = f"""
    Berdasarkan hasil analisis K-Means dengan jumlah cluster {k}, 
    diperoleh pemisahan kelompok pelanggan yang memiliki karakteristik berbeda 
    berdasarkan usia, pendapatan, dan tingkat pengeluaran. Visualisasi Before-Clustering 
    menunjukkan sebaran awal data, sedangkan hasil clustering memperlihatkan pemisahan 
    cluster yang lebih jelas. Elbow Method memperkuat pemilihan nilai K. Grafik rata-rata 
    cluster memberikan gambaran bahwa tiap kelompok memiliki perilaku belanja dan tingkat 
    pendapatan yang khas, sehingga dapat dimanfaatkan untuk strategi pemasaran yang tepat sasaran.
    """

    return render(request, 'clustering/kesimpulan.html', {
        "plot_awal": plot_awal,
        "elbow_plot": elbow_plot,
        "cluster_plot": cluster_plot,
        "summary_plot": summary_plot,
        "k": k,
        "summary": summary.to_html(classes="table table-bordered"),
        "kesimpulan_text": kesimpulan_text
    })

