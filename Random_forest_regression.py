from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import winsound
import os.path
import random
import pickle


class Node():
    # Constructor
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        # For decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red  # Variance reduction corresponding to split

        # For leaf node
        self.value = value  # Only valid for leaf nodes


class DecisionTreeRegressor():
    # Constructor
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None

        # Stopping conditions
        # Minimum amount of samples allowed in leaf_node
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.features = None

    def build_tree(self, dataset, curr_depth=0):
        # Recursive function to build the tree
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        best_split = {}

        # Split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(
                dataset, num_samples, num_features)
            # Check if information gain is positive (<0 indicates pure leaf node)
            if best_split["var_red"] > 0:
                # Recur left
                left_subtree = self.build_tree(
                    best_split["dataset_left"], curr_depth+1)
                # Recur right
                right_subtree = self.build_tree(
                    best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree, best_split["var_red"])

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_var_red = -float("inf")  # Infinitly large negative number

        # Loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            # Returns sorted unique elements of an array
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
                # Get current split with given threshold
                dataset_left, dataset_right, = self.split(
                    dataset, feature_index, threshold)
                # Check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -
                                                 1], dataset_left[:, -1], dataset_right[:, -1]
                    # Compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # Update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red

        if len(best_split) == 0:
            best_split["var_red"] = 0

        return best_split

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array(
            [row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array(
            [row for row in dataset if row[feature_index] > threshold])

        return dataset_left, dataset_right

    def variance_reduction(self, parent, l_child, r_child):
        weight_l = len(l_child)/len(parent)
        weight_r = len(r_child)/len(parent)
        reduction = np.var(parent) - (weight_l *
                                      np.var(l_child) + weight_r * np.var(r_child))

        return reduction

    def calculate_leaf_value(self, Y):
        val = np.mean(Y)

        return val

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=",
                  tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        # Function to train thr tree
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def make_predictions(self, x, tree):
        # Function to predict new dataset
        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_predictions(x, tree.left)
        else:
            return self.make_predictions(x, tree.right)

    def predict(self, X):
        if self.features != None:
            self.features = self.features[:-1]
            X = X[:, self.features]

        # Function to predict a single data point
        predictions = [self.make_predictions(x, self.root) for x in X]

        return predictions


class RandomForest():
    # Constructor
    def __init__(self, num_trees=64, num_sample_features=2, bootstrap_sample_size=50):
        self.forest = None
        self.num_trees = num_trees
        self.num_sample_features = num_sample_features
        self.bootstrap_sample_size = bootstrap_sample_size
        self.bootstrap_samples = None

    def generate_bootstrap_samples(self, X, Y):
        dataset = np.concatenate((X, Y), axis=1)
        num_samples, num_features = np.shape(X)

        bootstrap_samples = []  # Initialize an empty list to store the DataFrames

        for i in range(self.num_trees):
            # Feature indexes for single decision tree
            feature_indexes = random.sample(
                range(0, num_features), self.num_sample_features)
            feature_indexes.append(12)

            bootstrap_sample_data = []  # List to store the data of the current bootstrap sample

            for j in range(self.bootstrap_sample_size):
                # Select random sample id from the training dataset
                row_id = random.sample(range(0, num_samples), 1)

                # Append the row data of the current bootstrap sample to the list
                bootstrap_sample_data.append(dataset[row_id, feature_indexes])

            # Create a DataFrame for the current bootstrap sample
            bootstrap_sample_df = pd.DataFrame(
                np.vstack(bootstrap_sample_data))

            # Set the column names for the DataFrame based on the feature indexes
            column_names = [
                f"{feature_index}" for feature_index in feature_indexes]
            # column_names[-1] = "Y"
            bootstrap_sample_df.columns = column_names

            # Append the DataFrame to the list of bootstrap samples
            bootstrap_samples.append(bootstrap_sample_df)

        # Store the list of DataFrames as self.bootstrap_samples
        self.bootstrap_samples = bootstrap_samples

    def train_decision_trees(self, min_samples_split=3, max_depth=3):
        forest = np.empty(self.num_trees, dtype=DecisionTreeRegressor)

        for i in range(self.num_trees):
            data = self.bootstrap_samples[i]
            features = data.columns.to_list()
            features = [int(num) for num in features]

            _, X, _, Y = populate_data_arrays(
                None, data, "12")

            # X = self.bootstrap_samples[i, :, :-1]
            # Y = self.bootstrap_samples[i, :, -1][np.newaxis, :].T

            forest[i] = DecisionTreeRegressor(min_samples_split, max_depth)
            forest[i].features = features
            forest[i].fit(X, Y)

        return forest

    def fit(self, X, Y, min_samples_split=3, max_depth=3):
        self.generate_bootstrap_samples(X, Y)
        self.forest = self.train_decision_trees(min_samples_split, max_depth)

    def predict(self, X):
        if self.forest is None:
            print("[ERROR]: Random Forest regression model has not been trained.")
            return None
        else:
            pred_arr = np.empty((self.num_trees, X.shape[0]))

            for i in range(self.num_trees):
                pred_arr[i] = self.forest[i].predict(X)

            y_pred = np.empty(X.shape[0])
            for i in range(X.shape[0]):
                y_pred[i] = np.sum([pred_arr[:, i]], axis=1)/(self.num_trees)

            return y_pred


def import_data(neigh, clean, norm, stand, test, train_tree, train_forest, params):
    if clean or train_tree or train_forest:
        if test:
            filename = "Test_data/Test_" + str(neigh) + ".xlsx"
        else:
            filename = "Train_data/Train_" + str(neigh) + ".xlsx"
    elif norm:
        if test:
            filename = "Test_data/Test_norm_" + str(neigh) + ".xlsx"
        else:
            filename = "Norm_data/Norm_" + str(neigh) + ".xlsx"
    elif stand:
        if test:
            filename = "Test_data/Test_stand_" + str(neigh) + ".xlsx"
        else:
            filename = "Stand_data/Stand_" + str(neigh) + ".xlsx"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 "Resources", filename)

    data = pd.read_excel(data_path, sheet_name="Sheet1")
    df = pd.DataFrame(data, columns=params)

    print('[INFO]:', neigh, "DF loaded")
    print('[DF shape]:', df.shape)

    Y_label = 'SALE_PRICE_ESC'

    return df, Y_label


def import_features(neigh, test):
    if test:
        filename = "Test_data/Test_" + str(neigh) + ".xlsx"
    else:
        filename = "Train_data/Train_" + str(neigh) + ".xlsx"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 "Resources", filename)

    data = pd.read_excel(data_path, sheet_name="Sheet1")
    df = pd.DataFrame(data, columns=["DWEL_SIZE", "ERF_SIZE"])

    return df


def import_model(neigh, norm, stand, forest_flag):
    if norm:
        if forest_flag:
            filename = str(neigh) + "_norm.sav"
        else:
            filename = str(neigh) + "norm.sav"
    elif stand:
        if forest_flag:
            filename = str(neigh) + "_stand.sav"
        else:
            filename = str(neigh) + "_stand.sav"
    else:
        if forest_flag:
            filename = str(neigh) + ".sav"
        else:
            filename = str(neigh) + ".sav"

    if forest_flag:
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/RF_results/Models', neigh, filename)
    else:
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/DT_results/Models', neigh, filename)

    model = pickle.load(open(data_path, 'rb'))

    return model


def param_list(params, nr_param):
    param_new = []

    for i in range(nr_param):
        param_new.append(params[i])

    param_new.append(params[len(params)-1])

    return param_new


def kfold_cross_validation_random(df, params):
    count = 0

    shuffled_df = df.copy(deep=True)

    # Shuffle the rows randomly
    shuffled_df = shuffle(shuffled_df, random_state=42)
    shuffled_df.reset_index(drop=True, inplace=True)

    fold1 = pd.DataFrame(columns=params)
    fold2 = pd.DataFrame(columns=params)
    fold3 = pd.DataFrame(columns=params)
    fold4 = pd.DataFrame(columns=params)

    length = len(shuffled_df.index)

    while (count < length):
        row = shuffled_df.loc[count]
        fold1 = pd.concat([fold1, row.to_frame().T], ignore_index=True)
        count += 1

        if count == length:
            break

        row = shuffled_df.loc[count]
        fold2 = pd.concat([fold2, row.to_frame().T], ignore_index=True)
        count += 1

        if count == length:
            break

        row = shuffled_df.loc[count]
        fold3 = pd.concat([fold3, row.to_frame().T], ignore_index=True)
        count += 1

        if count == length:
            break

        row = shuffled_df.loc[count]
        fold4 = pd.concat([fold4, row.to_frame().T], ignore_index=True)
        count += 1

    return fold1, fold2, fold3, fold4


def split_train_test(fold1, fold2, fold3, fold4, iter=0):
    if iter == 0:
        df_train = pd.concat([fold1, fold2, fold3])
        df_test = fold4
    elif iter == 1:
        df_train = pd.concat([fold1, fold2, fold4])
        df_test = fold3
    elif iter == 2:
        df_train = pd.concat([fold1, fold4, fold3])
        df_test = fold2
    elif iter == 3:
        df_train = pd.concat([fold4, fold2, fold3])
        df_test = fold1

    df_train.reset_index(drop=True, inplace=True)
    return df_train, df_test


def populate_data_arrays(df_train, df_test, Y_label):
    X_test = np.zeros((df_test.shape[0], df_test.shape[1]-1))
    for i in range(df_test.shape[1]-1):
        for j in range(df_test.shape[0]):
            X_test[j, i] = float(df_test.at[j, df_test.columns[i]])

    y_test = np.zeros((df_test.shape[0], 1))
    for i in range(df_test.shape[0]):
        y_test[i, 0] = float(df_test.at[i, Y_label])

    if df_train is None:
        y_train = None
        X_train = None
    else:
        y_train = np.zeros((df_train.shape[0], 1))
        for i in range(df_train.shape[0]):
            y_train[i, 0] = float(df_train.at[i, Y_label])

        X_train = np.zeros((df_train.shape[0], df_train.shape[1]-1))
        for i in range(df_train.shape[1]-1):
            for j in range(df_train.shape[0]):
                X_train[j, i] = float(df_train.at[j, df_train.columns[i]])

    return X_train, X_test, y_train, y_test


def perc_error(y_test, y_pred):
    sum = 0
    for i in range(len(y_pred)):
        sum += ((abs(y_pred[i]-y_test[i]))/y_test[i]*100)

    return (sum/len(y_pred))


def train_depth_and_split(neigh_df, params, norm, stand):
    # Iterations
    depth_iter = 40
    split_iter = 40
    kfolds = 4

    # Arrays
    depth_arr = []
    split_arr = []
    acc_arr = []

    # Start and step
    min_samples_split = 1
    max_depth = 1
    split_step = 1
    depth_step = 1

    # Splitting dataset up into folds
    fold1, fold2, fold3, fold4 = kfold_cross_validation_random(
        neigh_df, params)

    for _ in tqdm(range(depth_iter), desc="Depth iteration"):
        min_samples_split = 1

        for _ in tqdm(range(split_iter), desc="Split iteration"):
            depth_arr.append(max_depth)
            split_arr.append(min_samples_split)
            XV_acc_arr = []

            for l in range(kfolds):
                Y_label = 'SALE_PRICE_ESC'

                # Kfold cross validation
                df_train, df_test = split_train_test(
                    fold1, fold2, fold3, fold4, iter=l)

                # Populate X and y matrices
                X_train, X_test, y_train, y_test = populate_data_arrays(
                    df_train, df_test, Y_label)

                # Scaling y_train
                if norm:
                    sc_X = MinMaxScaler()
                    sc_X.fit(X_train)
                    X_train = sc_X.transform(X_train)
                    X_test = sc_X.transform(X_test)
                elif stand:
                    sc_X = StandardScaler()
                    sc_X.fit(X_train)
                    X_train = sc_X.transform(X_train)
                    X_test = sc_X.transform(X_test)

                # Creating and training decision tree regressor model
                regressor = DecisionTreeRegressor(
                    min_samples_split=min_samples_split, max_depth=max_depth)
                regressor.fit(X_train, y_train)

                y_pred_test = regressor.predict(X_test)

                accuracy = perc_error(y_test, y_pred_test)

                XV_acc_arr.append(accuracy[0])

            acc_arr.append(np.mean(XV_acc_arr))

            min_samples_split += split_step

        max_depth += depth_step

    ind = acc_arr.index(min(acc_arr))
    opt_acc = acc_arr[ind]
    opt_split = split_arr[ind]
    opt_depth = depth_arr[ind]

    return split_arr, depth_arr, acc_arr, opt_split, opt_depth, opt_acc


def train_trees_and_features(neigh_df, min_samples_split, max_depth, params, neigh, backup_forest, norm, stand):
    # Iterations
    features_iter = 12
    features_start = 0
    trees_iter = 150
    kfolds = 4

    # Arrays
    features_arr = []
    trees_arr = []
    acc_arr = []

    # Start and step
    num_trees = 1
    num_sample_features = 1
    trees_step = 1
    features_step = 1

    # Splitting dataset up into folds
    fold1, fold2, fold3, fold4 = kfold_cross_validation_random(
        neigh_df, params)

    # Load backup
    if backup_forest > 1:
        filename = "Prelim_" + str(backup_forest) + "_results.xlsx"
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/RF_results/Models", neigh, "Prelimanary_results", filename)
        df_load = pd.read_excel(data_path)

        features_arr = df_load["Features"].values.tolist()
        trees_arr = df_load["Trees"].values.tolist()
        acc_arr = df_load["Accuracy"].values.tolist()

        num_sample_features = backup_forest
        features_start = backup_forest - 1

        print("[INFO]: Restored backup", filename)

    for _ in tqdm(range(features_start, features_iter), desc="Features iteration"):
        num_trees = 1

        # Backup results
        if num_sample_features > 1:
            df_save = pd.DataFrame(
                {"Features": features_arr, "Trees": trees_arr, "Accuracy": acc_arr})
            filename = "Prelim_" + str(num_sample_features) + "_results.xlsx"
            data_path = r''+os.path.join(os.path.dirname(__file__),
                                         "Resources/Model_data/RF_results/Models", neigh, "Prelimanary_results", filename)
            df_save.to_excel(data_path, index=False)
            print("[INFO]: Saved backup", filename)

        for _ in tqdm(range(trees_iter), desc="Trees iteration"):
            trees_arr.append(num_trees)
            features_arr.append(num_sample_features)
            XV_acc_arr = []

            for l in range(kfolds):
                Y_label = 'SALE_PRICE_ESC'

                # Kfold cross validation
                df_train, df_test = split_train_test(
                    fold1, fold2, fold3, fold4, iter=l)

                # Populate X and y matrices
                X_train, X_test, y_train, y_test = populate_data_arrays(
                    df_train, df_test, Y_label)

                bootstrap_sample_size = X_train.shape[0]

                # Scaling y_train
                if norm:
                    sc_X = MinMaxScaler()
                    sc_X.fit(X_train)
                    X_train = sc_X.transform(X_train)
                    X_test = sc_X.transform(X_test)
                elif stand:
                    sc_X = StandardScaler()
                    sc_X.fit(X_train)
                    X_train = sc_X.transform(X_train)
                    X_test = sc_X.transform(X_test)

                # Creating and training random forest regressor model
                forest = RandomForest(
                    num_trees, num_sample_features, bootstrap_sample_size)
                forest.fit(X_train, y_train, min_samples_split, max_depth)

                y_pred_test = forest.predict(X_test)

                accuracy = perc_error(y_test, y_pred_test)

                XV_acc_arr.append(accuracy[0])

            acc_arr.append(np.mean(XV_acc_arr))

            num_trees += trees_step

        num_sample_features += features_step

    ind = acc_arr.index(min(acc_arr))
    opt_acc = acc_arr[ind]
    opt_trees = trees_arr[ind]
    opt_features = features_arr[ind]

    return trees_arr, features_arr, acc_arr, opt_trees, opt_features, opt_acc


def train_model(neigh_df, min_samples_split, max_depth, num_trees, num_sample_features, forest_flag):
    Y_label = 'SALE_PRICE_ESC'

    # Populate X and y matrices
    _, X_test, _, y_test = populate_data_arrays(
        None, neigh_df, Y_label)

    if forest_flag:
        bootstrap_sample_size = X_test.shape[0]

        RF_regressor = RandomForest(
            num_trees, num_sample_features, bootstrap_sample_size)
        RF_regressor.fit(X_test, y_test, min_samples_split, max_depth)

        return RF_regressor
    else:
        DT_regressor = DecisionTreeRegressor(
            min_samples_split=min_samples_split, max_depth=max_depth)
        DT_regressor.fit(X_test, y_test)

        return DT_regressor


def plot_split_depth_opt(acc_arr, split_arr, depth_arr, norm, stand, neigh, save_spl_dep_plot):
    if norm:
        write_file_name1 = str(neigh) + "_RT_split_depth_norm"
    elif stand:
        write_file_name1 = str(neigh) + "_RT_split_depth_stand"
    else:
        write_file_name1 = str(neigh) + "_RT_split_depth"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/DT_results/Graphs/Optimising_split_depth', write_file_name1)

    split_arr = np.array(split_arr)
    depth_arr = np.array(depth_arr)
    acc_arr = np.array(acc_arr)

    split_arr = np.unique(split_arr)
    depth_arr = np.unique(depth_arr)

    X, Y = np.meshgrid(split_arr, depth_arr)
    Z = acc_arr.reshape(len(split_arr), len(depth_arr))

    # Plot the 3D surface
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Split and Depth Grid Search')
    ax.set_xlabel('Split')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Percentage error')

    if save_spl_dep_plot:
        plt.savefig(data_path, dpi=1500)

    plt.show()


def plot_trees_feature_opt(acc_arr, trees_arr, features_arr, norm, stand, neigh, save_tree_feat_plot):
    if norm:
        write_file_name1 = str(neigh) + "_RT_trees_feature_norm"
    elif stand:
        write_file_name1 = str(neigh) + "_RT_trees_feature_stand"
    else:
        write_file_name1 = str(neigh) + "_RT_trees_feature"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/RF_results/Graphs', write_file_name1)

    features_arr = np.unique(np.array(features_arr))
    trees_arr = np.unique(np.array(trees_arr))
    acc_arr = np.array(acc_arr)

    X, Y = np.meshgrid(trees_arr, features_arr)
    Z = acc_arr.reshape(len(features_arr), len(trees_arr))

    # Plot the 3D surface
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Trees and Features Grid Search')
    ax.set_xlabel('Trees')
    ax.set_ylabel('Sample features')
    ax.set_zlabel('Percentage error')

    if save_tree_feat_plot:
        plt.savefig(data_path, dpi=1500)

    plt.show()


def plot_predictions(y_test, y_pred, neigh, save_plt_pred, test):
    # Set style
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)

    
    plt.figure(figsize=(6.5, 5.6))
    # Plot model's predictions in red
    plt.scatter(y_test, y_pred, c="r", label="Predictions")

    if np.max(y_test) > np.max(y_pred):
        max = np.max(y_test)
    else:
        max = np.max(y_pred)

    if np.min(y_test) < np.min(y_pred):
        min = np.min(y_test)
    else:
        min = np.min(y_pred)
        
    # Plot the line y = x
    plt.plot([min-100000, max+100000], [min-100000, max+100000], c="b", label="y = x", linestyle="dashed")

    
    if test:
        title = "Actual VS Predicted Price - " + str(neigh) + " (Test)"
    else:
        title = "Actual VS Predicted Price - " + str(neigh) + " (Validation)"
        
    plt.title(title, y=1.01, fontsize=15.5)
    plt.xlabel("Actual price (R)", labelpad=10, fontsize=14.5)
    plt.ylabel("Predicted price (R)", labelpad=10, fontsize=14.5)
    plt.xlim([min, max + 10000])
    plt.ylim([min, max + 10000])

    # Show a legend
    plt.legend()
    plt.tight_layout()
    
    if save_plt_pred:
        if test:
            filename = str(neigh) + "_RF_test.png"
        else:
            filename = str(neigh) + "_RF_train.png"

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/RF_results/Models', neigh, filename)

        fig = plt.gcf()  # Get the current figure
        plt.savefig(data_path, dpi=150)
        print("[INFO]: ", str(neigh), "plot predictions saved")
    
    plt.show()


def save_to_file(X_train, y_train, y_pred_train, X_test, y_test, y_pred_test, model_code, forest_flag):
    y_test = y_test.flatten()
    y_train = y_train.flatten()

    # Create dataframe with predicted and test values
    df_train = pd.DataFrame({'DWG_EXT': X_train[:, 1], 'ERF_EXT': X_train[:, 1], 'GAR': X_train[:, 12],
                             'NO BEDS': X_train[:, 2], 'CON': X_train[:, 4], 'SEC': X_train[:, 7], 'Predicted value': y_pred_train,
                            'Real value': y_train, 'Model code': model_code})

    df_test = pd.DataFrame({'DWG_EXT': X_test[:, 1], 'ERF_EXT': X_test[:, 1], 'GAR': X_test[:, 12],
                            'NO BEDS': X_test[:, 2], 'CON': X_test[:, 4], 'SEC': X_test[:, 7], 'Predicted value': y_pred_test,
                            'Real value': y_test, 'Error percentage': ((abs(y_pred_test-y_test))/y_test*100), 'Model code': model_code})

    # Write dataframes to file
    if forest_flag == False:
        write_file_name = 'DT_results/DT_train_' + model_code + '.xlsx'
    else:
        write_file_name = 'RF_results/RF_train_' + model_code + '.xlsx'
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data', write_file_name)
    df_train.to_excel(data_path)

    if forest_flag == False:
        write_file_name = 'DT results/DT_test_' + model_code + '.xlsx'
    else:
        write_file_name = 'RF results/RF_test_' + model_code + '.xlsx'
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/RF_results', write_file_name)
    df_test.to_excel(data_path)

    print("[INFO]: Files saved successfully")


def save_trained_model(model, neigh, norm, stand, forest_flag):
    if norm:
        if forest_flag:
            filename = str(neigh) + "_norm.sav"
        else:
            filename = str(neigh) + "_norm.sav"
    elif stand:
        if forest_flag:
            filename = str(neigh) + "_stand.sav"
        else:
            filename = str(neigh) + "_stand.sav"
    else:
        if forest_flag:
            filename = str(neigh) + ".sav"
        else:
            filename = str(neigh) + ".sav"

    if forest_flag:
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/RF_results/Models', neigh, filename)
    else:
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/DT_results/Models', neigh, filename)

    # save the model to disk
    pickle.dump(model, open(data_path, 'wb'))


def save_predictions(y_test, y_pred_test, neigh, stand, norm, test, forest_flag, save_fusion_pred, features):
    y_test = y_test.flatten()
    df = pd.DataFrame({'Predicted value': y_pred_test, 'Real value': y_test,
                      'Percentage error': ((abs(y_pred_test-y_test))/y_test*100)})

    df["DWEL_SIZE"] = features["DWEL_SIZE"]
    df["ERF_SIZE"] = features["ERF_SIZE"]

    if norm:
        if forest_flag:
            filename = str(neigh) + "_forest_norm.xlsx"
        else:
            filename = str(neigh) + "_tree_norm.xlsx"
    elif stand:
        if forest_flag:
            filename = str(neigh) + "_forest_stand.xlsx"
        else:
            filename = str(neigh) + "_tree_stand.xlsx"
    else:
        if forest_flag:
            filename = str(neigh) + "_forest.xlsx"
        else:
            filename = str(neigh) + "_tree.xlsx"

    # Write dataframes to file
    if forest_flag:
        if test:
            filename2 = str(neigh) + "_test_RF.xlsx"
            data_path = r''+os.path.join(os.path.dirname(__file__),
                                         'Resources/Model_data/RF_results/Test_predictions', filename)
            data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                          'Resources/Model_data/Fusion_results/Data/Test', neigh, filename2)
        else:
            filename2 = str(neigh) + "_RF.xlsx"
            data_path = r''+os.path.join(os.path.dirname(__file__),
                                         'Resources/Model_data/RF_results/Validation_predictions', filename)
            data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                          'Resources/Model_data/Fusion_results/Data/Train', neigh, filename2)
    else:
        if test:
            data_path = r''+os.path.join(os.path.dirname(__file__),
                                         'Resources/Model_data/DT_results/Test_predictions', filename)
        else:
            data_path = r''+os.path.join(os.path.dirname(__file__),
                                         'Resources/Model_data/DT_results/Validation_predictions', filename)

    if forest_flag and save_fusion_pred:
        df.to_excel(data_path2)

    df.to_excel(data_path)


def provide_metrics(y_test, y_pred):
    # Flatten column vectors
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Compute Pearson coefficient
    corr, _ = pearsonr(y_test_flat, y_pred_flat)
    print("[Pearson]:", corr)
    
    # Compute MAE
    mae = np.mean(np.abs(y_test - y_pred))
    print("[MAE]:", mae)


def completion_noise():
    for _ in range(3):
        duration = 400
        freq = 700
        winsound.Beep(freq, duration)


def main():  # ID3 method
    # Extra variables
    neighbourhoods = ["Pinehurst", "Edgemead", "Brackenfell"]
    neigh_num = 0

    nr_param = 12
    params = ["ERF_SIZE", "DWEL_SIZE", "GAR_SIZE", "CPORT_SIZE", "NR_BEDS",
              "B_FIX_ONE", "B_FIX_TWO", "B_FIX_THREE", "B_FIX_FOUR", "CON",
              "QUAL", "VIEW", "SALE_PRICE_ESC"]
    params = param_list(params, nr_param)

    # Data type variables
    norm = False
    stand = False
    clean = False

    # Training variables
    train_tree = False
    train_forest = False
    backup_forest = 1  # Number of backup - One if no backup
    train_model_ = False
    vali_para = False
    test = False

    # Save variables
    save_spl_dep_plot = False
    save_tree_feat_plot = False
    save_model = False
    save_pred = False
    save_fusion_pred = False
    save_plt_pred = False

    # Plotting variables
    plot_split_depth = False
    plot_tree_features = False
    plt_pred = False
    metrics = False

    features = import_features(neighbourhoods[neigh_num], test)

    # Model parameters
    forest_flag = True
    min_samples_split = 26
    max_depth = 7
    num_trees = 85
    num_sample_features = 9

    # Training
    if train_tree:
        # Import data from a file
        neigh_df, Y_label = import_data(
            neighbourhoods[neigh_num], clean, norm, stand, test, train_tree, train_forest, params)

        print("[INFO]: Entered tree training")
        split_arr, depth_arr, acc_arr, opt_split, opt_depth, opt_acc = train_depth_and_split(
            neigh_df, params, norm, stand)

        min_samples_split = opt_split
        max_depth = opt_depth

        print("[INFO]: Optimal split and depth")
        print("       ", opt_split, "|", opt_depth, "|", opt_acc, "%")

        print("[INFO]: Tree training complete")

        completion_noise()

        if plot_split_depth:
            plot_split_depth_opt(acc_arr, split_arr, depth_arr,
                                 norm, stand, neighbourhoods[neigh_num], save_spl_dep_plot)

    if train_forest:
        print("[INFO]: Entered forest training")
        # Import data from a file
        neigh_df, Y_label = import_data(
            neighbourhoods[neigh_num], clean, norm, stand, test, train_tree, train_forest, params)

        trees_arr, features_arr, acc_arr, opt_trees, opt_features, opt_acc = train_trees_and_features(
            neigh_df, min_samples_split, max_depth, params, neighbourhoods[neigh_num], backup_forest, norm, stand)

        num_trees = opt_trees
        num_sample_features = opt_features

        print("[INFO]: Optimal trees and features")
        print("       ", opt_trees, "|", opt_features, "|", opt_acc, "%")

        print("[INFO]: Forest training complete")

        completion_noise()

        if plot_tree_features:
            plot_trees_feature_opt(acc_arr, trees_arr, features_arr,
                                   norm, stand, neighbourhoods[neigh_num], save_tree_feat_plot)

    train_tree = False
    train_forest = False

    # Import data from a file
    neigh_df, Y_label = import_data(
        neighbourhoods[neigh_num], clean, norm, stand, test, train_tree, train_forest, params)

    if train_model_:
        print("[INFO]: Entered model training")
        model = train_model(neigh_df, min_samples_split, max_depth,
                            num_trees, num_sample_features, forest_flag)

        print("[INFO]: Model training complete")

        if save_model:
            save_trained_model(
                model, neighbourhoods[neigh_num], norm, stand, forest_flag)

            print("[INFO]: Saved trained model successfully")

    # Validation
    if vali_para:
        print("[INFO]: Performing validation")

        _, X_test, _, y_test = populate_data_arrays(
            None, neigh_df, Y_label)

        if forest_flag:
            RF_regressor = import_model(
                neighbourhoods[neigh_num], norm, stand, forest_flag)

            y_pred_test = RF_regressor.predict(X_test)

        else:
            DT_regressor = import_model(
                neighbourhoods[neigh_num], norm, stand, forest_flag)

            y_pred_test = DT_regressor.predict(X_test)

        accuracy = perc_error(y_test, y_pred_test)
        print('[Accuracy]:', accuracy[0], '%')

        print("[INFO]: Validation complete")

        if save_pred:
            save_predictions(y_test, y_pred_test, neighbourhoods[neigh_num],
                             stand, norm, test, forest_flag, save_fusion_pred, features)

            print("[INFO]: Validation predictions saved successfully")

        if plt_pred:
            plot_predictions(y_test, y_pred_test, neighbourhoods[neigh_num], save_plt_pred, test)
            
        if metrics:
            provide_metrics(y_test, y_pred_test)

    # Test
    if test:
        print("[INFO]: Performing test")
        _, X_test, _, y_test = populate_data_arrays(
            None, neigh_df, Y_label)

        if forest_flag:
            RF_regressor = import_model(
                neighbourhoods[neigh_num], norm, stand, forest_flag)

            y_pred_test = RF_regressor.predict(X_test)

        else:
            DT_regressor = import_model(
                neighbourhoods[neigh_num], norm, stand, forest_flag)

            y_pred_test = DT_regressor.predict(X_test)

        accuracy = perc_error(y_test, y_pred_test)
        print('[Accuracy]:', accuracy[0], '%')

        print("[INFO]: Test complete")

        if save_pred:
            save_predictions(y_test, y_pred_test, neighbourhoods[neigh_num],
                             stand, norm, test, forest_flag, save_fusion_pred, features)

            print("[INFO]: Test predictions saved successfully")

        if plt_pred:
            plot_predictions(y_test, y_pred_test, neighbourhoods[neigh_num], save_plt_pred, test)
            
        if metrics:
            provide_metrics(y_test, y_pred_test)

    completion_noise()

    print("[INFO]: Algorithm complete")


if __name__ == "__main__":
    main()
