import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs # 生成聚集分布的一组点
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STKaiti', 'Arial Unicode MS']  # 优先使用苹方，其次是楷体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

X,y_true = make_blobs(n_samples=300,centers=3, cluster_std=2,random_state=42)

fig,ax = plt.subplots(2,figsize=(10,10))
ax[0].scatter(X[:,0],X[:,1],c="gray",label="原始数据")
ax[0].set_title("原始数据")


kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
centers = kmeans.cluster_centers_
y_kmeans = kmeans.predict(X)
ax[1].scatter(X[:,0],X[:,1],c=y_kmeans,label="聚类数据")
ax[1].set_title("聚类数据")
ax[1].scatter(centers[:,0],centers[:,1],c='red')
plt.show()