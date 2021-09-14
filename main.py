import sys
from mainGUI import *
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from fpgrowth.utils import make_prediction, read_pickle
from matplotlib import pyplot as plt
import pickle as pkl
import math
from math import exp

class MyForm(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.spinBox.setValue(10)
        self.ui.spinBox_2.setValue(10)
        self.ui.mdiArea.addSubWindow(self.ui.NeuralNetwork)
        self.ui.mdiArea.addSubWindow(self.ui.AssociationRule)
        self.ui.mdiArea.addSubWindow(self.ui.HierarchicalClustering)
        self.ui.pushButton.clicked.connect(self.train_nn)
        self.ui.pushButton_2.clicked.connect(self.run_clustering)
        self.ui.pushButton_3.clicked.connect(self.create_nn_csv)
        self.ui.ar_button.clicked.connect(self.suggest_assoc)

    def train_nn(self):
        global X_train
        global y_train
        global bias1, bias2, weight1, weight2

        # NN Parameters
        train_data = X_train
        train_target = y_train
        learning_rate = 0.1
        num_input_neurons = len(X_train[0])
        num_hidden_neurons = self.ui.spinBox.value()
        num_output_neurons = y_train.max() + 1

        # initialize weights
        weight1 = np.random.default_rng().standard_normal(size=(num_input_neurons, num_hidden_neurons),
                                                          dtype=np.float32) - 0.5
        bias1 = np.random.default_rng().standard_normal(size=num_hidden_neurons, dtype=np.float32) - 0.5
        weight2 = np.random.default_rng().standard_normal(size=(num_hidden_neurons, num_output_neurons),
                                                          dtype=np.float32) - 0.5
        bias2 = np.random.default_rng().standard_normal(size=num_output_neurons, dtype=np.float32) - 0.5

        # setting the loop parameters
        error_diff = float("inf")
        iteration = 0
        curr_error = 0
        prev_error = 0
        stopping_error = 0.000001
        train_data_length = len(train_data)
        max_iteration = self.ui.spinBox_2.value()
        all_output_signals = np.zeros((20, 20), np.float32)
        np.fill_diagonal(all_output_signals, 1.0)

        import time
        start_time = time.time()
        # main loop
        while error_diff > stopping_error and iteration < max_iteration:
            running_error = 0
            for index_row in range(train_data_length):
                # adjust input and output signals
                input_signals = np.array(train_data[index_row], dtype=np.float32)
                output_signals = all_output_signals[train_target.iloc[index_row]]

                # calculate feed forward from input layer
                z_inj = np.add(input_signals @ weight1, bias1)

                # activation function
                zj = 1.0 / (1.0 + np.exp(-z_inj))

                # calculate feed forward from hidden layer
                y_ink = np.add(zj @ weight2, bias2)

                # activation function
                yk = 1.0 / (1.0 + np.exp(-y_ink))

                # calculate the error
                output_error = output_signals - yk
                small_delta_k = np.multiply(np.multiply(output_error, yk), (1 - yk))
                running_error += output_error @ output_error

                # back propagate the error
                small_delta_in_j = small_delta_k @ weight2.T
                small_delta_j = np.multiply(np.multiply(zj, small_delta_in_j), (1 - zj))

                # update weights and biases
                bias1 += np.multiply(learning_rate, small_delta_j)
                weight1 += np.multiply(np.mat(input_signals).T * np.mat(small_delta_j), learning_rate)
                bias2 += np.multiply(learning_rate, small_delta_k)
                weight2 += np.multiply(np.mat(zj).T * np.mat(small_delta_k), learning_rate)

                # accumulating Total Squared Error
                curr_error += running_error

            # printing loop info
            error_diff = abs(prev_error - curr_error)
            if (iteration % 2 == 0) or (error_diff < stopping_error):
                print('Error: ', curr_error, ', epoch: ', iteration, ', error difference: ', error_diff)
                print(">> %s second" % (time.time() - start_time))

            iteration += 1
            prev_error = curr_error
            curr_error = 0

        # finishing
        print(">> %s seconds" % (time.time() - start_time))
        self.ui.label_2.setText("Done Training!")

    def create_nn_csv(self):
        # set input and output variables
        test_set = vectors_test
        output_nn = [None] * len(test_set)

        # main loop to produce the result
        for iter1 in range(len(test_set)):
            # adjust input signals
            input_test = vectors_test[iter1]

            # calculate the feed forward
            z_inj = np.add(input_test @ weight1, bias1)
            zj = 1.0 / (1.0 + np.exp(-z_inj))
            y_ink = np.add(zj @ weight2, bias2)
            yk = 1.0 / (1.0 + np.exp(-y_ink))

            # record the result to output array
            output_nn[iter1] = np.argmax(yk)

        # create the dataframe to write csv
        dframe = pd.DataFrame(data=output_nn)
        y2 = dframe.apply(le.inverse_transform)
        item_id_dframe = pd.DataFrame(itemID_test)
        df_output = pd.concat([item_id_dframe, y2], axis=1)
        df_output.columns = ['id', 'cuisine']
        df_output.to_csv('output_submission.csv', index=False)

        # finishing
        print('writing Predictions to csv file complete')
        self.ui.label_2.setText("Done creating submission!")

    def suggest_assoc(self):
        ingredients = self.ui.ar_text_box.text().split(",")
        suggestions = make_prediction(ingredients=set(ingredients), top_n_suggestions=5,
                                      rules_path="./fpgrowth/rules.pkl")
        suggestion_list = [','.join(suggestion[0]) for suggestion in suggestions]
        self.ui.ar_list_widget.clear()
        if suggestions:
            self.ui.ar_list_widget.addItems(suggestion_list)
        else:
            self.ui.ar_list_widget.addItem("No associated items found")

    def run_clustering(self):
        data = X_train[:1000]
        k = 10
        print(f'=====\nRunning Agglomerative Hierarchical Clustering algorithm with k={k}...')
        # Start with every element in its own cluster
        clusters = {}
        print(f'# Elements={len(data)}')
        for i, x in enumerate(data):
            clusters[single_tuple(i)] = x
        # Run ACH
        global merges
        merges = []
        clusters = cluster(data, clusters, k)
        print(f'-----\nGenerated {len(clusters)} final clusters:')
        for i, elements in enumerate(clusters.keys()):
            distribution = {}
            print(f'\nCluster #{i+1}/{len(clusters.keys())}:')
            for i in range(len(elements)):
                #print(f'[{elements[i]}]={meal_region[elements[i]]}: {itemList[elements[i]]}')
                region = meal_region[elements[i]]
                if distribution.get(region) == None:
                    distribution[region] = 1
                else:
                    distribution[region] += 1
            for region in distribution:
                print(f'{region} = \t{distribution[region]/len(elements) * 100}%')

        # Cached the results as running the whole algorithm is time-consuming
        with open('ahc/cache.pkl', 'rb') as f:
            Z = pkl.load(f)
        # Make the dendrogram without a plot to make a label function
        from scipy.cluster.hierarchy import dendrogram
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Distance')
        plt.ylabel('Cuisine (Element ID)')
        D = dendrogram(
            Z,
            no_plot=True,
            p = 25,
            truncate_mode = 'lastp',
        )
        leaves = D["leaves"]
        # Get string labels for elements
        labels = []
        for index in leaves:
            region = meal_region[index]
            labels.append(region)
        # A function to substitue the string labels
        temp = {leaves[index]: labels[index] for index in range(len(leaves))}
        def llf(index):
            return "{} (ID#{})".format(temp[index], index)
        # Make the dendrogram
        dendrogram(
            Z,
            orientation='right',
            distance_sort='descending',
            leaf_label_func=llf,
            p = 25,
            truncate_mode = 'lastp',
            show_contracted=True
        )
        plt.show()

def cluster(data, centroids, k):
    # Runtime monitoring
    import time
    # Cluster until k clusters are made
    print(f'Clustering with k={k} with {len(centroids)} initial clusters...')
    next_index = len(centroids)
    # Track merges to build dendrogram
    merges = []
    while (len(centroids) > k):
        start = time.time()
        clusters = {}
        cluster_keys = {}
        distances = {}
        idx = 0;
        keys = list(centroids.keys())
        # Get the distances between all clusters
        for i in range(len(keys) - 1):
            ikey = keys[i]
            for j in range(i + 1, len(keys)):
                jkey = keys[j]
                cluster_keys[idx] = [ikey, jkey]
                clusters[idx] = ikey + jkey
                distances[idx] = euclidean_distance(centroids[ikey], centroids[jkey])
                idx += 1
        # Get smallest distance
        sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
        indices = list(sorted_distances.keys())
        index = indices[0]
        distance = sorted_distances[index]
        # Calculate the new centroid
        centroid = compute_centroid(data, clusters[index])
        centroids[clusters[index]] = centroid
        merges.append([clusters[index], distance, len(cluster_keys[index])])
        # Remove the old centroid
        for cluster_key in cluster_keys[index]:
            centroids.pop(cluster_key)
        print(f'Found {len(centroids)} clusters in {time.time() - start} seconds')
    return centroids

# Go through the elements in the cluster and average the values across all their indices
def compute_centroid(dataset, clusters):
    attributes = len(dataset[0])
    centroid = [0.0] * attributes
    for idx in range(len(clusters)):
        element_index = clusters[idx]
        element = dataset[element_index]
        for i in range(attributes):
            centroid[i] += float(element[i])
    for i in range(attributes):
        centroid[i] /= len(clusters)
    return centroid

# Return a tuple with a single value
def single_tuple(value):
    return ((value,))

# Get the euclidian distance between the two points
def euclidean_distance(x, y):
    # Only measure the indices for which at least one of the points has a value
    # In other words ignore elements that are missing for both to save computation time and space
    total_points = {}
    for i, key in enumerate(x):
        if key > 0:
            if i not in list(total_points.keys()):
                total_points[i] = [0, 0]
            total_points[i][0] = 1
        if y[i] > 0:
            if i not in list(total_points.keys()):
                total_points[i] = [0, 0]
            total_points[i][1] = 1
    total_keys = list(total_points.keys())
    total_keys.sort()
    # Calculate the distance
    result = 0
    for key in total_keys:
        result += pow(float(total_points[key][0]) - float(total_points[key][1]), 2)
    result = result ** 0.5
    return result


def setup_data():
    print("Setting the data...")
    # Reading the Yummly dataset
    with open('./train.json', encoding='utf-8', errors='replace') as dataset:
        data = dataset.read()[3:-3]
        data = data.split("},")
        data.append("temp")
        rows = []
        for row in data[:-1]:
            row = row + "}"
            rows.append(json.loads(row))

    # split the json file info into id, region, and ingredients
    global itemList
    itemList = []
    itemID = []
    global meal_region
    meal_region = []
    for row in rows:
        m = ""
        itemID.append(row['id'])
        meal_region.append(row['cuisine'])
        for each1 in row['ingredients']:
            # replace space in the ingredients with underscore
            each1 = each1.replace(' ', '_')
            m += each1 + ' '
        itemList.append(m)

    # binaryzing train data
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(itemList).toarray()

    # List of ingredients for association rule suggestion
    global rules
    rules = []

    # Read the ingredient list from file
    rules = read_pickle("./fpgrowth/rules.pkl")

    # Now, processing test data
    with open('./test.json', encoding='utf-8', errors='replace') as testset:
        data = testset.read()[3:-3]
        data = data.split("},")
        data.append("temp")
        meals_test = []
        for row in data[:-1]:
            row = row + "}"
            meals_test.append(json.loads(row))

    # split the json file info into id, region, and ingredients
    itemList_test = []
    global itemID_test
    itemID_test = []
    meal_cuisine_test = []
    for each in meals_test:
        m = ""
        itemID_test.append(each['id'])
        for row in each['ingredients']:
            row = row.replace(' ', '_')
            m += row + ' '
        itemList_test.append(m)

    # binaryzing test data with train data vocabulary
    vectorizer2 = CountVectorizer(vocabulary=vectorizer.vocabulary_)
    global vectors_test
    vectors_test = vectorizer2.fit_transform(itemList_test).toarray()

    # creating training dataframe
    global le
    le = preprocessing.LabelEncoder()
    data = {"cuisine": meal_region}
    dataframe = pd.DataFrame(data)
    y = dataframe.apply(le.fit_transform)
    global X_train
    global y_train
    X_train = vectors
    y_train = y['cuisine']


global X_train
global y_train
global vectors_test
global bias1, bias2, weight1, weight2
global le
global itemID_test
global itemList

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    setup_data()
    myapp = MyForm()
    myapp.run_clustering()
    #myapp.show()
    #sys.exit(app.exec_())
    sys.exit()
