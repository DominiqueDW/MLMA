from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from ann_visualizer.visualize import ann_viz
from keras.utils import plot_model
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from statistics import mean
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
import os.path
import pickle
import os

# Disable or enable tensorflow warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', patience=0, verbose=0, restore_best_weights=False):
        super(CustomEarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_epoch = None
        self.best_value = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if current_value < self.best_value:
            self.best_value = current_value
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print(
                            f"Restoring model weights from epoch {self.best_epoch}")
                    self.model.set_weights(self.best_weights)


def import_data(neigh, test):
    if test:
        filename1 = str(neigh) + "_test_MPR.xlsx"
        filename2 = str(neigh) + "_test_RF.xlsx"
        filename3 = str(neigh) + "_test_SVR.xlsx"
        filename4 = str(neigh) + "_test_NN.xlsx"
        print("[INFO]: Loading test data for", neigh)
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Test", neigh, filename1)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_MPR = pd.DataFrame(
            data, columns=["Predicted value", "Real value", "DWEL_SIZE", "ERF_SIZE"])

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Test", neigh, filename2)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_RF = pd.DataFrame(data, columns=["Predicted value"])

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Test", neigh, filename3)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_SVR = pd.DataFrame(data, columns=["Predicted value"])

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Test", neigh, filename4)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_NN = pd.DataFrame(data, columns=["Predicted value"])
    else:
        filename1 = str(neigh) + "_MPR.xlsx"
        filename2 = str(neigh) + "_RF.xlsx"
        filename3 = str(neigh) + "_SVR.xlsx"
        filename4 = str(neigh) + "_NN.xlsx"
        print("[INFO]: Loading training data for", neigh)
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Train", neigh, filename1)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_MPR = pd.DataFrame(
            data, columns=["Predicted value", "Real value", "DWEL_SIZE", "ERF_SIZE"])

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Train", neigh, filename2)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_RF = pd.DataFrame(data, columns=["Predicted value"])

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Train", neigh, filename3)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_SVR = pd.DataFrame(data, columns=["Predicted value"])

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     "Resources/Model_data/Fusion_results/Data/Train", neigh, filename4)
        data = pd.read_excel(data_path, sheet_name="Sheet1")
        df_NN = pd.DataFrame(data, columns=["Predicted value"])

    print('[DF shape]:', df_MPR.shape)

    Y_label = "Real value"

    return df_MPR, df_RF, df_SVR, df_NN, Y_label


def import_model(neigh, norm, stand):
    if norm:
        model_name = str(neigh) + "_norm"
    elif stand:
        model_name = str(neigh) + "_stand"
    else:
        model_name = str(neigh)

    filename = "Resources/Model_data/Fusion_results/Models/" + \
        str(neigh) + "/" + model_name
    filename2 = "Resources/Model_data/Fusion_results/Models/" + \
        str(neigh) + "/scaler_y.pkl"
    filename3 = "Resources/Model_data/Fusion_results/Models/" + \
        str(neigh) + "/scaler_X.pkl"

    NN_model = tf.keras.models.load_model(filename)
    sc_y = pickle.load(open(filename2, "rb"))
    sc_X = pickle.load(open(filename3, "rb"))

    return NN_model, sc_X, sc_y


def merge_dfs(df_MPR, df_RF, df_SVR, df_NN):
    # Concatenate the "Predicted" columns
    predicted_concatenated = pd.concat(
        [df_MPR['Predicted value'], df_RF['Predicted value'], df_SVR['Predicted value'], df_NN['Predicted value']], axis=1)

    features = pd.concat([df_MPR['DWEL_SIZE'], df_MPR['ERF_SIZE']], axis=1)

    real_column = df_MPR['Real value']

    # Combine the concatenated "Predicted" columns with the selected "Real" column
    df_neigh = pd.concat(
        [predicted_concatenated, features, real_column], axis=1)

    # Set column names for the new data frame
    df_neigh.columns = ['Pred_MPR', 'Pred_RF',
                        'Pred_SVR', 'Pred_NN', 'DWEL_SIZE', 'ERF_SIZE', 'Real value']

    print('[DF shape]:', df_neigh.shape)

    return df_neigh


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


def compile_model(NN_model, loss_func, opt_func, lr, momentum):
    # Set loss function parameters
    if loss_func == "MAE":
        if opt_func == "SGD":
            NN_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
                             metrics=["mse"])
        elif opt_func == "Adam":
            NN_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                             metrics=["mse"])
        elif opt_func == "rmsprop":
            NN_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
                             metrics=["mse"])
    elif loss_func == "MSE":
        if opt_func == "SGD":
            NN_model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum),
                             metrics=["mse"])
        elif opt_func == "Adam":
            NN_model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                             metrics=["mse"])
        elif opt_func == "rmsprop":
            NN_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
                             metrics=["mse"])


def fit_model(NN_model, X_train, y_train, X_test, y_test, epochs, optimise_NN):
    if optimise_NN:
        # EarlyStopping Callback
        callback = CustomEarlyStopping(
            monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)

        history = NN_model.fit(np.asarray(X_train), np.asarray(
            y_train), epochs=epochs, verbose=0, validation_data=(np.asarray(X_test), np.asarray(y_test)), callbacks=[callback])

    else:
        callback = CustomEarlyStopping(
            monitor='val_loss', patience=5, verbose=0, restore_best_weights=True)
        callback.best_epoch = None

        history = NN_model.fit(np.asarray(X_train), np.asarray(
            y_train), epochs=epochs, verbose=0)

    return history, callback.best_epoch


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
    plt.plot([min-100000, max+100000], [min-100000, max+100000],
             c="b", label="y = x", linestyle="dashed")

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
            filename = str(neigh) + "_NN_fusion_test.png"
        else:
            filename = str(neigh) + "_NN_fusion_train.png"

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/Fusion_results/Models', neigh, filename)

        fig = plt.gcf()  # Get the current figure
        plt.savefig(data_path, dpi=150)
        print("[INFO]:", str(neigh), "plot predictions saved")

    plt.show()


def plot_history(history, loss_func):
    # Set style
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14}
    matplotlib.rc('font', **font)

    plt.plot(history.history['loss'], label="Training loss")
    plt.plot(history.history['val_loss'], label="Validation loss")
    plt.legend()
    plt.title(f"{loss_func} Loss VS Epochs")
    plt.ylabel("Loss", fontsize=15, labelpad=8)
    plt.xlabel("Epochs", labelpad=5)
    plt.tight_layout()
    plt.show()


def save_predictions(y_test, y_pred_test, neigh, stand, norm, test, features):
    df = pd.DataFrame({'Predicted value': y_pred_test, 'Real value': y_test,
                      'Percentage error': ((abs(y_pred_test-y_test))/y_test*100)})

    df["DWEL_SIZE"] = features["DWEL_SIZE"]
    df["ERF_SIZE"] = features["ERF_SIZE"]

    if norm:
        filename = str(neigh) + "_norm.xlsx"
    elif stand:
        filename = str(neigh) + "_stand.xlsx"
    else:
        filename = str(neigh) + ".xlsx"

    # Write dataframes to file
    if test:
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/Fusion_results/Test_predictions', filename)
    else:
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/Fusion_results/Validation_predictions', filename)

    df.to_excel(data_path)


def create_NN(num_layers, num_nodes, act_funcs, loss_func, method, lr, momentum):
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    # Initialise sequential model
    NN_model = tf.keras.Sequential()

    # Add hidden layers
    for i in range(num_layers):
        NN_model.add(tf.keras.layers.Dense(units=num_nodes, activation=act_funcs[i], use_bias=True,
                                           kernel_initializer="glorot_uniform", bias_initializer="zeros"))

    # Add output layer
    NN_model.add(tf.keras.layers.Dense(
        1, "linear", True, "glorot_uniform", "zeros"))

    compile_model(NN_model, loss_func,
                  method, lr, momentum)

    return NN_model


def select_opt_NN(neigh):
    filename = "Resources/Model_data/Fusion_results/Optimisation_tests/" + \
        str(neigh) + "/results.txt"

    with open(filename, "r") as file:
        content = file.readlines()

    # Parsing each line to extract accuracy and storing the corresponding line
    accuracy_line_map = {}
    for line in content:
        # Splitting each line based on commas to get individual key-value pairs
        parts = line.split(", ")
        for part in parts:
            # Extracting accuracy value
            if "MAPE" in part:
                accuracy = float(part.split(": ")[1])
                accuracy_line_map[accuracy] = line

    # Getting the line with the lowest accuracy
    min_accuracy = min(accuracy_line_map.keys())
    optimal_line = accuracy_line_map[min_accuracy]

    return optimal_line


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


def get_structure(struct):
    if struct == 1:
        num_nodes = 8
        act_funcs = ["tanh", "tanh", "tanh"]
        num_layers = 2
    elif struct == 2:
        num_nodes = 8
        act_funcs = ["sigmoid", "sigmoid", "sigmoid"]
        num_layers = 2
    elif struct == 3:
        num_nodes = 8
        act_funcs = ["ReLU", "ReLU", "ReLU"]
        num_layers = 2
    elif struct == 4:
        num_nodes = 15
        act_funcs = ["tanh", "tanh", "tanh"]
        num_layers = 2
    elif struct == 5:
        num_nodes = 15
        act_funcs = ["sigmoid", "sigmoid", "sigmoid"]
        num_layers = 2
    elif struct == 6:
        num_nodes = 15
        act_funcs = ["ReLU", "ReLU", "ReLU"]
        num_layers = 2
    elif struct == 7:
        num_nodes = 8
        act_funcs = ["tanh", "tanh", "tanh"]
        num_layers = 3
    elif struct == 8:
        num_nodes = 8
        act_funcs = ["sigmoid", "sigmoid", "sigmoid"]
        num_layers = 3
    elif struct == 9:
        num_nodes = 8
        act_funcs = ["ReLU", "ReLU", "ReLU"]
        num_layers = 3
    elif struct == 10:
        num_nodes = 15
        act_funcs = ["tanh", "tanh", "tanh", "tanh"]
        num_layers = 3
    elif struct == 11:
        num_nodes = 15
        act_funcs = ["sigmoid", "sigmoid", "sigmoid"]
        num_layers = 3
    elif struct == 12:
        num_nodes = 15
        act_funcs = ["ReLU", "ReLU", "ReLU", "ReLU"]
        num_layers = 3
    elif struct == 13:
        num_nodes = 8
        act_funcs = ["tanh", "tanh", "tanh", "tanh"]
        num_layers = 4
    elif struct == 14:
        num_nodes = 8
        act_funcs = ["sigmoid", "sigmoid", "sigmoid", "sigmoid"]
        num_layers = 4
    elif struct == 15:
        num_nodes = 8
        act_funcs = ["ReLU", "ReLU", "ReLU", "ReLU"]
        num_layers = 4

    return num_nodes, act_funcs, num_layers


def main():
    # Extra variables
    neighbourhoods = ["Pinehurst", "Edgemead", "Brackenfell"]
    neigh_num = 0
    params = ['Pred_MPR', 'Pred_RF',
              'Pred_SVR', 'Pred_NN', 'DWEL_SIZE', 'ERF_SIZE', 'Real value']

    # Data type variables
    norm = False
    stand = False

    # Train variables
    optimise_NN = False
    select_NN = False
    train_NN = False  # Only train optimal configuration (Save ones sc_y)
    vali_para = False
    test = False

    # Save variables
    save_model = False
    save_graph_model = False
    save_pred = False
    save_plt_pred = False
    save_plt_bars = False

    # Plot variables
    plt_pred = False
    plt_history = False
    plt_graph_model = False
    plt_NN_diagram = False
    plt_final_results = True
    metrics = False

    # Model hyperparameters
    kfolds = 4
    momentum = 0
    epochs = 100

    lr = 0.01
    loss_func = "MSE"
    method = "rmsprop"

    # Structure parameters
    struct = 1
    num_nodes, act_funcs, num_layers = get_structure(struct)

    features = import_features(neighbourhoods[neigh_num], test)

    # Import data
    df_MPR, df_RF, df_SVR, df_NN, Y_label = import_data(
        neighbourhoods[neigh_num], test)

    df_neigh = merge_dfs(df_MPR, df_RF, df_SVR, df_NN)

    # Optimisation
    if optimise_NN:
        print("[INFO]: Entered neural network optimisation")

        # Grid search values
        lr_values = [0.001, 0.01, 0.1]
        loss_funcs = ["MAE", "MSE"]
        opt_methods = ["SGD", "Adam", "rmsprop"]
        data_types = ["norm", "stand"]

        for i in range(1, 16, 1):
            struct = i
            num_nodes, act_funcs, num_layers = get_structure(struct)

            lr_arr = []
            loss_arr = []
            method_arr = []
            data_arr = []
            acc_arr = []
            pear_arr = []
            epochs_arr = []

            for data_type in data_types:

                if data_type == "norm":
                    norm = True
                    stand = False
                else:
                    norm = False
                    stand = True

                for lr in lr_values:
                    for loss_func in loss_funcs:
                        for method in opt_methods:
                            XV_acc_arr = []
                            XV_pear_arr = []
                            XV_epochs_arr = []

                            # Splitting dataset up into folds
                            fold1, fold2, fold3, fold4 = kfold_cross_validation_random(
                                df_neigh, params)

                            for l in range(kfolds):
                                # Kfold cross validation
                                df_train, df_test = split_train_test(
                                    fold1, fold2, fold3, fold4, iter=l)

                                # Populate X and y matrices
                                X_train, X_test, y_train, y_test = populate_data_arrays(
                                    df_train, df_test, Y_label)

                                # Scaling y_train
                                if norm:
                                    sc_X = MinMaxScaler()
                                    sc_y = MinMaxScaler()
                                    X_train = sc_X.fit_transform(X_train)
                                    X_test = sc_X.transform(X_test)
                                    y_train = sc_y.fit_transform(y_train)
                                    y_test = sc_y.transform(y_test)
                                elif stand:
                                    sc_X = StandardScaler()
                                    sc_y = StandardScaler()
                                    X_train = sc_X.fit_transform(X_train)
                                    X_test = sc_X.transform(X_test)
                                    y_train = sc_y.fit_transform(y_train)
                                    y_test = sc_y.transform(y_test)

                                # Initialise NN
                                NN_model = create_NN(
                                    num_layers, num_nodes, act_funcs, loss_func, method, lr, momentum)

                                try:
                                    history, epoch = fit_model(
                                        NN_model, X_train, y_train, X_test, y_test, epochs, optimise_NN)

                                    # plot_history(history, loss_func)

                                    # Make predictions for model
                                    y_pred = NN_model.predict(X_test)

                                    # Unscaling y_pred
                                    if norm or stand:
                                        y_pred = sc_y.inverse_transform(y_pred)
                                        y_test = sc_y.inverse_transform(y_test)

                                    accuracy = perc_error(y_test, y_pred)

                                    y_test_flat = y_test.flatten()
                                    y_pred_flat = y_pred.flatten()
                                    # Calculate Pearson correlation
                                    corr, _ = pearsonr(
                                        y_test_flat, y_pred_flat)

                                except:
                                    print("[INFO]: NN training failed")
                                    accuracy = [0]
                                    corr = -10
                                    epoch = 0

                                XV_acc_arr.append(accuracy[0])
                                XV_pear_arr.append(corr)
                                XV_epochs_arr.append(epoch)

                            accuracy = round(mean(XV_acc_arr), 3)
                            pcorr = round(mean(XV_pear_arr), 3)
                            avg_epoch = mean(XV_epochs_arr)

                            lr_arr.append(lr)
                            loss_arr.append(loss_func)
                            method_arr.append(method)
                            data_arr.append(data_type)
                            acc_arr.append(accuracy)
                            pear_arr.append(pcorr)
                            epochs_arr.append(avg_epoch)

            acc_ind = acc_arr.index(min(acc_arr))
            opt_acc = acc_arr[acc_ind]
            opt_p = pear_arr[acc_ind]
            opt_lr = lr_arr[acc_ind]
            opt_loss = loss_arr[acc_ind]
            opt_method = method_arr[acc_ind]
            opt_data = data_arr[acc_ind]
            opt_epoch = epochs_arr[acc_ind]

            pear_ind = pear_arr.index(max(pear_arr))
            opt_p2 = pear_arr[pear_ind]
            opt_acc2 = acc_arr[pear_ind]
            opt_lr2 = lr_arr[pear_ind]
            opt_loss2 = loss_arr[pear_ind]
            opt_method2 = method_arr[pear_ind]
            opt_data2 = data_arr[pear_ind]
            opt_epoch2 = epochs_arr[pear_ind]

            opt_config = (f"struct: {struct}, "
                          f"lr: {opt_lr}, loss_func: {opt_loss}, "
                          f"opt_func: {opt_method}, data_type: {opt_data}, "
                          f"Epochs: {opt_epoch}, MAPE: {opt_acc}, r: {opt_p}\n")

            opt_config2 = (f"struct: {struct}, "
                           f"lr: {opt_lr2}, loss_func: {opt_loss2}, "
                           f"opt_func: {opt_method2}, data_type: {opt_data2}, "
                           f"Epochs: {opt_epoch2}, MAPE: {opt_acc2}, r: {opt_p2}\n")

            print("[INFO]: Neural network optimisation complete")

            filename = "Resources/Model_data/Fusion_results/Optimisation_tests/" + \
                str(neighbourhoods[neigh_num]) + "/results.txt"

            f = open(filename, "a")
            f.write(opt_config)
            f.close()

            f = open(filename, "a")
            f.write(opt_config2)
            f.close()

    if select_NN:
        opt_config = select_opt_NN(neighbourhoods[neigh_num])
        print("[INFO]: Optimal configuration")
        print(opt_config)

        print("[INFO]: Model selection complete")

    if train_NN:
        print("[INFO]: Entered neural network training")

        # Populate X and y matrices
        _, X_train, _, y_train = populate_data_arrays(
            None, df_neigh, Y_label)

        # Initialise NN
        NN_model = create_NN(
            num_layers, num_nodes, act_funcs, loss_func, method, lr, momentum)

        # Scaling X_train and y_train
        if norm:
            sc_X = MinMaxScaler()
            sc_y = MinMaxScaler()
            X_train = sc_X.fit_transform(X_train)
            y_train = sc_y.fit_transform(y_train)
        elif stand:
            sc_X = StandardScaler()
            sc_y = StandardScaler()
            X_train = sc_X.fit_transform(X_train)
            y_train = sc_y.fit_transform(y_train)
        else:
            sc_X = None
            sc_y = None

        history = fit_model(NN_model, X_train, y_train,
                            None, None, epochs, optimise_NN)

        # Make predictions for model
        y_pred = NN_model.predict(X_train)

        # Unscaling y_pred
        if norm or stand:
            y_pred = sc_y.inverse_transform(y_pred)
            y_train = sc_y.inverse_transform(y_train)

        accuracy = perc_error(y_train, y_pred)
        print("[INFO]: Accuracy:", accuracy[0], "%")

        if save_graph_model:
            if norm:
                model_name = str(
                    neighbourhoods[neigh_num]) + "_norm"
            elif stand:
                model_name = str(
                    neighbourhoods[neigh_num]) + "_stand"
            else:
                model_name = str(neighbourhoods[neigh_num])

            file_name = "Resources/Model_data/Fusion_results/Graphical_models/" + model_name
            ann_viz(NN_model, view=plt_graph_model, filename=file_name)

            print("[INFO]: Graphical neural network model saved successfully")

        if save_model:
            if norm:
                model_name = str(neighbourhoods[neigh_num]) + "_norm"
            elif stand:
                model_name = str(neighbourhoods[neigh_num]) + "_stand"
            else:
                model_name = str(neighbourhoods[neigh_num])

            file_name = "Resources/Model_data/Fusion_results/Models/" + \
                str(neighbourhoods[neigh_num]) + "/" + model_name
            filename2 = "Resources/Model_data/Fusion_results/Models/" + \
                str(neighbourhoods[neigh_num]) + "/scaler_y.pkl"
            filename3 = "Resources/Model_data/Fusion_results/Models/" + \
                str(neighbourhoods[neigh_num]) + "/scaler_X.pkl"

            NN_model.save(file_name)
            pickle.dump(sc_y, open(filename2, "wb"))
            pickle.dump(sc_X, open(filename3, "wb"))

            print("[INFO]: Neural network model saved successfully")

        if plt_history:
            plot_history(history, loss_func)

        if plt_pred:
            plot_predictions(
                y_train, y_pred, neighbourhoods[neigh_num], save_plt_pred, test)

        print("[INFO]: Completed neural network training")

    # Validation
    if vali_para:
        print("[INFO]: Performing validation")

        # Populate X and y matrices
        _, X_train, _, y_train = populate_data_arrays(
            None, df_neigh, Y_label)

        NN_model, sc_X, sc_y = import_model(
            neighbourhoods[neigh_num], norm, stand)

        if norm or stand:
            X_train = sc_X.transform(X_train)

        # Computing model predictions
        y_pred = NN_model.predict(X_train)

        # Unscaling y_pred
        if norm or stand:
            y_pred = sc_y.inverse_transform(y_pred)

        accuracy = perc_error(y_train, y_pred)

        print('[Accuracy]:', accuracy, '%')

        print("[INFO]: Validation complete")

        if save_pred:
            y_pred = y_pred.flatten()
            y_train = y_train.flatten()
            save_predictions(
                y_train, y_pred, neighbourhoods[neigh_num], stand, norm, test, features)

            print("[INFO]: Validation predictions saved successfully")

        if plt_pred:
            plot_predictions(
                y_train, y_pred, neighbourhoods[neigh_num], save_plt_pred, test)

        if metrics:
            provide_metrics(y_train, y_pred)

    # Test
    if test:
        print("[INFO]: Performing test")

        # Populate X and y matrices
        _, X_test, _, y_test = populate_data_arrays(
            None, df_neigh, Y_label)

        NN_model, sc_X, sc_y = import_model(
            neighbourhoods[neigh_num], norm, stand)

        if norm or stand:
            X_test = sc_X.transform(X_test)

        y_pred_test = NN_model.predict(X_test)

        if norm or stand:
            y_pred_test = sc_y.inverse_transform(y_pred_test)

        accuracy = perc_error(y_test, y_pred_test)

        print('[Accuracy]:', accuracy, '%')

        print("[INFO]: Test complete")

        if save_pred:
            y_pred_test = y_pred_test.flatten()
            y_test = y_test.flatten()
            save_predictions(
                y_test, y_pred_test, neighbourhoods[neigh_num], stand, norm, test, features)

            print("[INFO]: Test predictions saved successfully")

        if plt_NN_diagram:
            # Block visualization
            plot_model(NN_model, to_file="NN_models\model_1.png",
                       show_shapes=True)

        if plt_pred:
            plot_predictions(y_test, y_pred_test,
                             neighbourhoods[neigh_num], save_plt_pred, test)

        if metrics:
            provide_metrics(y_test, y_pred_test)

    if plt_final_results:
        # Set style
        font = {'family': 'Times New Roman',
                'weight': 'normal',
                'size': 12}
        matplotlib.rc('font', **font)

        # Create the bar plot with T-shaped caps for the whiskers
        fig, ax = plt.subplots()

        # Model names
        models = ["MPR", "RF", "SVR", "NN", "Fusion NN"]

        # Different colors for each bar
        colors = ['dodgerblue', 'green', 'red', 'purple', 'orange']

        # Accuracies for each model
        if neighbourhoods[neigh_num] == "Pinehurst":
            accuracies = [95.1, 95.8, 95.6, 95.3, 96.1]
        elif neighbourhoods[neigh_num] == "Edgemead":
            accuracies = [91.2, 90.3, 91.0, 92.0, 92.7]
        else:
            accuracies = [91.9, 92.6, 92.5, 93.0, 94.2]

        # Upper and lower bounds for whiskers
        if neighbourhoods[neigh_num] == "Pinehurst":
            # RF best, fusion next
            upper_bounds = [24.6, 16.7, 20.6, 22.7, 14.5]
            lower_bounds = [0.3, 0.0, 0.1, 0.3, 0.0]
        elif neighbourhoods[neigh_num] == "Edgemead":
            upper_bounds = [40.0, 49.0, 40.2, 44.4, 31.0]
            lower_bounds = [0.1, 0.3, 0.1, 0.0, 0.0]
        else:
            upper_bounds = [31.8, 21.2, 26.5, 30.3, 18.1]
            lower_bounds = [0.0, 0.2, 0.2, 0.0, 0.2]

        upper_bounds = [100 - x for x in upper_bounds]
        lower_bounds = [100 - x for x in lower_bounds]

        # Create bars
        bars = ax.bar(models, accuracies, color=colors,
                      alpha=0.7, align='center')

        colors2 = ["black", "black", "black", "black", "black"]

        # Add error bars with T-shaped caps
        for i, (accuracy, lower, upper) in enumerate(zip(accuracies, lower_bounds, upper_bounds)):
            ax.plot([i, i], [lower, upper], color='black')
            ax.plot([i-0.1, i+0.1], [lower, lower], color='black')
            ax.plot([i-0.1, i+0.1], [upper, upper], color='black')
            # Add text labels for accuracies on top of the bars
            ax.text(i-0.21, accuracy - 5,
                    f"{accuracy}", ha='center', color=colors2[i])
            # Add whisker upper and lower values on top of the whiskers
            ax.text(i, lower + 1, f"{lower}", ha='center')
            ax.text(i, upper - 5, f"{upper}", ha='center')

        # Adding labels and title
        ax.set_ylim(0, 110)
        ax.set_xlabel('Machine learning models', labelpad=10, fontsize=14)
        ax.set_ylabel('(100-MAPE) (%)', labelpad=5, fontsize=14)
        title = 'Accuracy Comparison of ML Models - ' + \
            str(neighbourhoods[neigh_num])
        ax.set_title(title, fontsize=16)

        plt.tight_layout()

        if save_plt_bars:
            filename = str(
                neighbourhoods[neigh_num]) + "_bargraph_test.png"

            data_path = r''+os.path.join(os.path.dirname(__file__),
                                         'Resources/Model_data/Fusion_results/Plots', filename)

            fig = plt.gcf()  # Get the current figure
            plt.savefig(data_path, dpi=150)

            print("[INFO]: ", str(neighbourhoods[neigh_num]),
                  "plot predictions saved")

        plt.show()


if __name__ == "__main__":
    main()
