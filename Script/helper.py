import itertools

def decode(labels):
    encoding = 'utf-8'

    # decode from byte to string
    labels = [str(x, encoding) for x in labels]
    label_decoded = [x.replace('_', ' ') for x in labels]

    return label_decoded

def get_labels_idx(keys, raw_labels):
    # key: category
    # value: index

    labels = {}

    for i in range(len(raw_labels)):
        for k in keys:
            if k in raw_labels[i]:
                if k not in labels:
                    labels[k] = [i]
                else:
                    labels[k].append(i)

    return labels

def get_clusters_dict(labels):
    # key: label
    # value: indices of images
    clusters = {}

    for idx, label in enumerate(labels):
        if label not in clusters:
            # The label is seen for first time, create a new list.
            clusters[label] = [idx]
        else:
            clusters[label].append(idx)

    return clusters

# Create label pairs

def create_label_pairs(labels):

    label_pairs = {}

    for key, value in labels.items():
        label_pairs[key] = list(itertools.combinations(value, 2))

    label_pairs_concat = []

    for key, value in label_pairs.items():
        label_pairs_concat += value

    return label_pairs_concat
