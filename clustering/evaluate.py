import os

class Evaluate:

    def __init__(self, label_names):

        self.label_names = label_names


    def get_true_labels(self, path):

        files = []
        files = [f for f in sorted(os.listdir(path))]

        num_images = [len(os.listdir(path + i)) for i in files]

        self.true_label = []

        for i in range(len(files)):
            self.true_label.append((files[i], num_images[i]))

        return self.true_label

    def compute_metric(self):

        # matching labels in same cluster
        true_positive = []
        # non matching labels in same clusters
        false_positive = []
        # non matching labels in different clusters
        false_negative = []

        for i in range(len(self.label_names)):

            # Count of the number of true labels in each cluster
            true_positive.append(self.label_names[i].count(self.true_label[i][0]))

            # size of each cluster minus true positive
            false_positive.append(len(self.label_names[i]) - self.label_names[i].count(self.true_label[i][0]))

            # size of each true label - true positive
            # computes the missing images in each true label
            false_negative.append(self.true_label[i][1] - self.label_names[i].count(self.true_label[i][0]))

        tp = sum(true_positive)
        fp = sum(false_positive)
        fn = sum(false_negative)

        self.precision = tp/(tp+fp)
        self.recall = tp/(tp+fn)
        self.f_measure = 2*((self.precision*self.recall)/(self.precision+self.recall))

        return self.precision, self.recall, self.f_measure