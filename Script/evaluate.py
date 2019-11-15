# F-measure

def evaluate(true_labels, cluster_labels):

    true_positive = list(set(true_labels).intersection(cluster_labels))
    false_positive = list(set(cluster_labels) - set(true_labels))
    false_negative = list(set(true_labels) - set(cluster_labels))

    TP = len(true_positive)
    FP = len(false_positive)
    FN = len(false_negative)

    precision = round(TP/(TP+FP), 3)

    recall = round(TP/(TP+FN), 3)

    f_measure = round(2*((precision*recall)/(precision+recall)), 3)

    return f_measure, precision, recall

def evaluate_average(true_label_pairs, num_clusters, num_iter, algo):

    cumulative_f_measure = 0
    cumulative_precision = 0
    cumulative_recall = 0
    cumulative_runtime = 0

    for i in range(num_iter):

        if algo == "Kmeans":

            start = time.time()

            kmeans = KMeans(n_clusters = num_clusters).fit(X)

            runtime = round((time.time() - start), 3)

            k_means_clusters = get_clusters_dict(kmeans.labels_)
            cluster_label_pairs = create_label_pairs(k_means_clusters)

        elif algo == "HAC":
            start = time.time()

            clustering = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=None).fit(X)

            runtime = round((time.time() - start), 3)

            hac_clusters = get_clusters_dict(clustering.labels_)
            cluster_label_pairs = create_label_pairs(hac_clusters)

        elif algo == "Spectral":

            start = time.time()

            clustering = SpectralClustering(n_clusters=num_clusters).fit(X)

            runtime = round((time.time() - start), 3)

            spectral_cluster = get_clusters_dict(clustering.labels_)
            cluster_label_pairs = create_label_pairs(spectral_cluster)

        elif algo == "GMM":
            start = time.time()

            gmm_labels = GaussianMixture(n_components=num_clusters, init_params='kmeans').fit_predict(X)

            runtime = round((time.time() - start), 3)

            gmm_clusters = get_clusters_dict(gmm_labels)

            cluster_label_pairs = create_label_pairs(gmm_clusters)

        elif algo == "Birch":

            start = time.time()

            brc = Birch(n_clusters=num_clusters, compute_labels=True).fit(X)

            birch_labels = brc.predict(X)

            runtime = round((time.time() - start), 3)

            birch_clusters = get_clusters_dict(birch_labels)

            cluster_label_pairs = create_label_pairs(birch_clusters)


        f_measure, precision, recall = evaluate(true_label_pairs, cluster_label_pairs)

        cumulative_f_measure += f_measure
        cumulative_precision += precision
        cumulative_recall += recall
        cumulative_runtime += runtime

    avg_f_measure = round(cumulative_f_measure/num_iter, 3)
    avg_precision = round(cumulative_precision/num_iter, 3)
    avg_recall = round(cumulative_recall/num_iter, 3)
    avg_runtime = round(cumulative_runtime/num_iter, 3)

    print("{} Average F-Measure: {}".format(algo, avg_f_measure))
    print("{} Average Precision: {}".format(algo, avg_precision))
    print("{} Average Recall: {}".format(algo, avg_recall))
    print("{} Average Runtime: {}".format(algo, avg_runtime))
