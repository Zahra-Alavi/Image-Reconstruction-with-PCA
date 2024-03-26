

import numpy as np
import matplotlib.pyplot as plt


import util
# TODO: change cluster_5350 to cluster if you do the extra credit
from cluster import *

######################################################################
# helper functions
######################################################################

def build_face_image_points(X, y):
    """
    Translate images to (labeled) points.

    Parameters
    --------------------
        X     -- numpy array of shape (n,d), features (each row is one image)
        y     -- numpy array of shape (n,), targets

    Returns
    --------------------
        point -- list of Points, dataset (one point for each image)
    """

    n,d = X.shape

    images = collections.defaultdict(list) # key = class, val = list of images with this class
    for i in range(n):
        images[y[i]].append(X[i,:])

    points = []
    for face in images:
        count = 0
        for im in images[face]:
            points.append(Point(str(face) + '_' + str(count), face, im))
            count += 1

    return points


def generate_points_2d(N, seed=1234):
    """
    Generate toy dataset of 3 clusters each with N points.

    Parameters
    --------------------
        N      -- int, number of points to generate per cluster
        seed   -- random seed

    Returns
    --------------------
        points -- list of Points, dataset
    """
    np.random.seed(seed)

    mu = [[0,0.5], [1,1], [2,0.5]]
    sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]

    label = 0
    points = []
    for m,s in zip(mu, sigma):
        label += 1
        for i in range(N):
            x = util.random_sample_2d(m, s)
            points.append(Point(str(label)+'_'+str(i), label, x))

    return points


######################################################################
# main
######################################################################

def main():
    
    
   #============================================================
   # Load and display data
   #============================================================

    X, y = util.get_lfw_data()    # load LFW dataset 
   
    util.plot_gallery(X[:12])   # Displaying the first 12 images as a gallery  
    
    
    for i in range(5):          # Displaying 5 sample images individually
        util.show_image(X[i], size=(50, 37)) 
    
    average_face = np.mean(X, axis=0).reshape(50, 37)    # Displaying "average" face
    util.show_image(average_face, size=(50, 37))
   
    
   #============================================================
   # run PCA and display top-12 eigenfaces, then reconstruct images 
   #============================================================
   
    U, mu = util.PCA(X)
    util.plot_gallery([util.vec_to_image(U[:,i]) for i in range(12)])
    

    for l in [1, 10, 50, 100, 500, 1850]:   # for each value of l (number of eigenfaces), apply PCA, reconstruct, and plot gallery
        Z, Ul = util.apply_PCA_from_Eig(X, U, l, mu)
        X_rec = util.reconstruct_from_PCA(Z, Ul, mu)
        util.plot_gallery(X_rec[:12])
        
        
   #============================================================
   # Clustering faces:
   # Run kMeans and kMedoids on the 4 selected classes, report min, max, and avg scores over 10 trials
   #============================================================                        
        
    np.random.seed(1234)

    X1, y1 = util.limit_pics(X, y, [4, 6, 13, 16], 40)  # Considering only four individuals (4, 6, 13, 16)
    points = build_face_image_points(X1, y1)
    
    
    def clusterscores(X, n_clusters, init):
        
        k_means_scores = []
        k_medoids_scores = []
        
    
        for _ in range(10):
                clustering_kmeans = kMeans(X, k=n_clusters, init=init)
                k_means_scores.append(clustering_kmeans.score())
              
                clustering_kmedoids = kMedoids(X, k=n_clusters, init=init)
                k_medoids_scores.append(clustering_kmedoids.score())
                
        avg_k_means = np.mean(k_means_scores)
        avg_k_medoids = np.mean(k_medoids_scores)
           
        return k_means_scores, k_medoids_scores , avg_k_means , avg_k_medoids
            
    
    k_means_scores, k_medoids_scores , avg_k_means , avg_k_medoids = clusterscores(points, n_clusters=4, init = "random")
    
    
    min_k_means = np.min(k_means_scores)
    max_k_means = np.max(k_means_scores)
    
    print("K-means:" , avg_k_means, min_k_means, max_k_means)

    min_k_medoids = np.min(k_medoids_scores)
    max_k_medoids = np.max(k_medoids_scores)
    print("K-medoids", avg_k_medoids, min_k_medoids, max_k_medoids)
    
    

# =============================================================================
# #### Uncomment to see the the effect of clustering on this subset of data  #######
#     points = build_face_image_points(X2, y2)
#     
#      
#     k_means_scores, k_medoids_scores , avg_k_means , avg_k_medoids = clusterscores(points, n_clusters=2, init = "random")
#     
#     
#     min_k_means = np.min(k_means_scores)
#     max_k_means = np.max(k_means_scores)
#     
#     print("K-means:" , avg_k_means, min_k_means, max_k_means)
# 
#     min_k_medoids = np.min(k_medoids_scores)
#     max_k_medoids = np.max(k_medoids_scores)
#     print("K-medoids", avg_k_medoids, min_k_medoids, max_k_medoids)
#     
# =============================================================================

 
   
    X2, y2 = util.limit_pics(X, y, [4, 13], 40)   # Considering only two individuals (4 and 13)
    
    U2, mu2 = util.PCA(X2) 
    

    ls = []
    
    for num in range(1, 41 + 1):  
     
    # checking condition
        if num % 2 != 0:
            ls.append(num)
    
    means_scores = []
    medoids_scores = []
    
    for l in ls:

        Z2, Ul = util.apply_PCA_from_Eig(X2, U, l, mu2)
        
        points = build_face_image_points(Z2, y2)
        
        *_, avg_k_means , avg_k_medoids = clusterscores(points, n_clusters=2, init = "cheat")
        
        means_scores.append(avg_k_means)
        medoids_scores.append(avg_k_medoids)
        
        
    # plot clustering score (for both kMeans and kMedoids) as a function of l   
    
    plt.figure()
    plt.plot(ls, means_scores, label="K-means")
    plt.plot(ls, medoids_scores, label="K-medoids")
    plt.xlabel("Number of Principal Components (l)")
    plt.ylabel("Clustering Score")
    plt.title("Clustering Score vs. Number of Principal Components")
    plt.legend()
    plt.grid(True)
    plt.show()
        
   #============================================================
   # determine "most discriminative" and "least discriminative" pairs of images
   # by finding the min and max clustering scores for each unique pair of individuals
   #============================================================  
    
   
    np.random.seed(1234)
   
    min_scores = []
    max_scores = []
    pairs = []
    
   
    
    for i in range(17):
        for j in range(i + 1, 17):
              
                pairs.append([i,j])
                newX, newy = util.limit_pics(X, y, [i, j], 40)
                points = build_face_image_points(newX, newy)
                k_means_scores, k_medoids_scores , avg_k_means , avg_k_medoids = clusterscores(points, n_clusters=2, init = "random")
                min_k_means = np.min(k_means_scores)
                max_k_means = np.max(k_means_scores)
                min_scores.append(min_k_means)
                max_scores.append(max_k_means)
               
    
    print(pairs, min_scores, max_scores)
    
    index = np.array(min_scores).argmax()
    print(max(min_scores), "max of min_scores:", pairs[index])
    
    index = np.array(max_scores).argmin()
    print(min(max_scores),"min of max scores", pairs[index])
    
    
    
    for i in [2,10,8,14]:
        Xi, yi = util.limit_pics(X, y, [i], 40)
        print("individual:", i)
        util.plot_gallery(Xi[:12])
    
    


if __name__ == "__main__":
    main()
