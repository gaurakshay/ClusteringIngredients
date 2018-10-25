#!usr/bin/env python
import sys
# Import libraries.
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import KMeans
from string import whitespace
import matplotlib.pyplot as plt
import numpy as np

def import_clean():
    # Import the file and convert it to a format
    # that is ready for vectorization and clustering.
    # There would definitely be a more elegent way to do it.
    t0 = time()
    file_ = open('srep00196-s2.csv', 'r')
    meals = file_.readlines()
    # Series of transformations to remove the underscores, commas and the digits
    # Remove white spaces.
    tmp = [meal.translate(None, whitespace) for meal in meals]
    tmp = [meal.split(',') for meal in tmp]  # Split by comma
    # Remove the integers at the end of each lin
    tmp = [meal[:-1] for meal in tmp]
    tmp = [' '.join(meal) for meal in tmp]  # Join the split string together
    tmp = [meal.split('_') for meal in tmp]  # Split by underscore.
    tmp = [' '.join(meal) for meal in tmp]  # Join the split string.
    meals = tmp[4:]  # Remove the first four lines not containing any ingredients.
    print ('Total time taken : {:0.3f}s. '.format(time() - t0)
           + 'Some of the terms of the files (total terms in file are {}): ' 
           .format(len(meals)))
    print meals[:15]
    return meals

def vectorize_file(input_file):
    # Vectorize the text. I don't believe stemming is
    # required here because I don't think any verbs are involved.
    # All the ingredients would be nouns.
    # min_df is set to 300 so that PCA can work. PCA fails if kept below 300.
    print 'Performing TfIdf transform'
    t0 = time()
    # tfidf_vectorizer = TfidfVectorizer(min_df = 300, max_df=600, max_features=800)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(input_file)
    print ('Transformation complete! Total time taken : {:0.3f}s'.format(time() - t0))
    # print 'type of matrix: "' + str(type(tfidf_matrix)) + '"'
    # print 'Shape of the sparse matrix: "' + str(tfidf_matrix.shape) + '"'
    return tfidf_vectorizer, tfidf_matrix

def cluster_kmean(clusters, input_matrix):
    # KMeans clustering.
    number_of_clusters = clusters
    print 'Preparing predictor...'
    t0 = time()
    km = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(input_matrix)
    # clusters = km.labels_.tolist()
    print ('Total time taken : {:0.3f}s'.format(time() - t0))
    # clusters = km.labels_
    # print type(clusters)
    # centroids = km.cluster_centers_
    # print centroids
    # print len(centroids)
    return km


def get_top_terms(number_of_clusters, vectorizer, km):
    terms = vectorizer.get_feature_names()
    # print type(terms)
    # print "Top terms per cluster:"
    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[::-1]
    # print order_centroids
    # print len(order_centroids)
    for i in range(number_of_clusters):
        print ("  Cluster {} :".format(i))
        # Use numpy array notation to get the terms that are in the centroid
        for ind in order_centroids[i, :5]:
            # print the terms associated with that term
            print('    {}'.format(terms[ind]))


def get_clusters(km, m, row_dict):
    labels = km.predict(m)
    clusters = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(row_dict[n])
        else:
            clusters[item] = [row_dict[n]]
        n += 1
    for item in clusters:
        print "cluster: ", item
        for i in clusters[item][:5]:
            print i


def plot_centroids(km):
    # Plot the centroids. Looking at how close the
    # clusters are, maybe I would need to tune the number
    # of clusters.

    centroids = km.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker="x", s=150, linewidths=2, zorder=10)
    plt.show()


def reduce_dimensions():
    # Reducing the dimensions for successfull plotting.
    print 'Reducing vector dimensions'
    pca = PCA(n_components=2)
    reduced_matrix_pca = pca.fit_transform(count_matrix.toarray())
    print 'New vector dimensions: '
    print reduced_matrix_pca.shape

    # km1 = KMeans(n_clusters=number_of_clusters)
    # %time km1.fit(reduced_matrix_pca)


def plot_datapoints():
    # Plot the datapoints
    x_min = reduced_matrix_pca[:, 0].min() + 1
    x_max = reduced_matrix_pca[:, 0].max() - 1
    y_min = reduced_matrix_pca[:, 1].min() + 1
    y_max = reduced_matrix_pca[:, 1].max() - 1
    h = 0.1
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.plot(reduced_matrix_pca[:, 0], reduced_matrix_pca[:, 1], 'k.', markersize=2)
    centroids = km.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1], marker="x",
               s=150, linewidths=5, zorder=10)
    pl.title('Plotting many many meals after count vectorization')
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    pl.show()


def try_plot(n, data):
    reduced_data = TruncatedSVD(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n, n_init=1, max_iter=100)
    kmeans.fit(reduced_data)
    h = 0.2
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    #plt.imshow(Z, interpolation='nearest',
    #           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    #           cmap=plt.cm.Paired,
    #           aspect='auto', origin='lower')
    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='r', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show(block=False)


def try_predict(vectorizer, km, new_data, old_matrix, old_data):
    tfidf_matrix = vectorizer.transform(new_data)
    result = km.predict(tfidf_matrix)
    labels = km.predict(old_matrix)
    clusters = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(old_data[n])
        else:
            clusters[item] = [old_data[n]]
        n += 1
    print ('Some of the closest matching dishes for your input are: ')
    for i in clusters[result[0]][:5]:
        print i
    return result


if __name__ == '__main__':
    print ('Importing the file and cleaning it...')
    n_clusters = 0
    meals = import_clean()
    print
    vectorizer, matrix = vectorize_file(meals)
    proceed = raw_input('Plot the chart? [y/n]: ')
    if proceed == 'y':
        clusters = raw_input('How many clusters do you want? ')
        try:
            n_clusters = int(clusters)
        except ValueError:
            print ('Incorrect input. Exiting.')
            sys.exit()
        try_plot(n_clusters, matrix)
    loop_var = True
    not_clustered = True
    while loop_var:
        user_input = raw_input('Enter 1 to search, 2 to exit: ')
        if user_input == '2':
            loop_var = False
        else:
            if not_clustered:
                if not n_clusters:
                    n_clusters = 6
                km = cluster_kmean(n_clusters, matrix)
                not_clustered = False
            input_ingrdnt = raw_input('Enter your ingredients in comma-separated format: ')
            input_array = input_ingrdnt.split(',')
            input_array = [x.strip() for x in input_array]
            input_array = ' '.join(input_array)
            input_ingrdnt = [input_array]
            predicted_centroid = try_predict(vectorizer, km, input_ingrdnt, matrix, meals)
