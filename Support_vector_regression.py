from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from statistics import mean
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import winsound
import os.path
import pickle


def import_data(neigh, clean, norm, stand, test, opt_C_eps, params):
    if clean or opt_C_eps:
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


def import_model(neigh, kernel, degree, test_norm, test_stand):
    if test_norm:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_norm.sav"
    elif test_stand:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_stand.sav"
    else:
        filename = str(neigh) + "_" + str(kernel) + "_" + str(degree) + ".sav"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/SVR_results/Models', neigh, filename)
    data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                        'Resources/Model_data/SVR_results/Models', neigh, "scaler.pkl")

    SVR_regressor = pickle.load(open(data_path, 'rb'))
    sc_y = pickle.load(open(data_path2, "rb"))

    return SVR_regressor, sc_y


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


def train_C_and_epsilon(neigh_df, degree, kernel, gamma, shrinking, norm, stand, params):
    # Iterations
    C_iter = 100
    eps_iter = 30
    kfolds = 4

    # Arrays
    C_arr = []
    eps_arr = []
    acc_arr = []

    # Start and step
    epsilon = 0
    C = 0.1
    eps_step = 0.1
    C_step = 1

    # Splitting dataset up into folds
    fold1, fold2, fold3, fold4 = kfold_cross_validation_random(
        neigh_df, params)

    for _ in tqdm(range(len(C_iter)), desc="C iteration"):
        epsilon = 0

        for _ in tqdm(range(eps_iter), desc="Epsilon iteration"):
            C_arr.append(C)
            eps_arr.append(epsilon)
            XV_acc_arr = []

            for l in range(kfolds):
                Y_label = 'SALE_PRICE_ESC'

                # Kfold cross validation
                df_train, df_test = split_train_test(
                    fold1, fold2, fold3, fold4, iter=l)

                # Populate X and y matrices
                X_train, X_test, y_train, y_test = populate_data_arrays(
                    df_train, df_test, Y_label)

                # Creating regressor model
                SVR_regressor = SVR(kernel=kernel, degree=degree,
                                    gamma=gamma, C=C, epsilon=epsilon, shrinking=shrinking)

                # Scaling y_train
                if norm:
                    sc_X = MinMaxScaler()
                    sc_y = MinMaxScaler()
                    sc_X.fit(X_train)
                    sc_y.fit(y_train)
                    X_train = sc_X.transform(X_train)
                    X_test = sc_X.transform(X_test)
                    y_train = sc_y.transform(y_train)
                elif stand:
                    sc_X = StandardScaler()
                    sc_y = StandardScaler()
                    sc_X.fit(X_train)
                    sc_y.fit(y_train)
                    X_train = sc_X.transform(X_train)
                    X_test = sc_X.transform(X_test)
                    y_train = sc_y.transform(y_train)
                y_train_flat = y_train.flatten()

                # Training regressor model
                SVR_regressor.fit(X_train, y_train_flat)

                # Computing model predictions
                y_pred_test = SVR_regressor.predict(X_test)

                if norm or stand:
                    y_pred_test = y_pred_test.reshape(-1, 1)
                    y_pred_test = sc_y.inverse_transform(y_pred_test)

                accuracy = perc_error(y_test, y_pred_test)

                XV_acc_arr.append(accuracy[0])

            acc_arr.append(mean(XV_acc_arr))

            epsilon += eps_step

    ind = acc_arr.index(min(acc_arr))
    opt_acc = acc_arr[ind]
    opt_C = C_arr[ind]
    opt_eps = eps_arr[ind]

    return eps_arr, C_arr, acc_arr, opt_C, opt_eps, opt_acc


def train_w_and_b(neigh_df, degree, kernel, gamma, shrinking, norm, stand, opt_eps, opt_C):
    Y_label = 'SALE_PRICE_ESC'

    # Populate X and y matrices
    _, X_train, _, y_train = populate_data_arrays(
        None, neigh_df, Y_label)

    # Creating regressor model
    SVR_regressor = SVR(kernel=kernel, degree=degree,
                        gamma=gamma, C=opt_C, epsilon=opt_eps, shrinking=shrinking)

    # Scaling y_train
    if norm:
        sc_y = MinMaxScaler()
        y_train = sc_y.fit_transform(y_train)
    elif stand:
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)
    else:
        sc_y = None
    y_train_flat = y_train.flatten()

    # Training regressor model
    SVR_regressor.fit(X_train, y_train_flat)

    return SVR_regressor, sc_y


def plot_C_epsilon_opt(acc_arr, C_arr, eps_arr, save_C_eps_plot, norm, stand, neigh, degree, opt_C, opt_eps, opt_acc):
    if norm:
        write_file_name1 = str(neigh) + "_SVR_C_eps_degree_" + \
            str(degree) + "_norm"
    elif stand:
        write_file_name1 = str(neigh) + "_SVR_C_eps_degree_" + \
            str(degree) + "_stand"
    else:
        write_file_name1 = str(neigh) + "_SVR_C_eps_degree_" + \
            str(degree)

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/SVR_results/Graphs/Optimising_C_eps', write_file_name1)

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 12}
    matplotlib.rc('font', **font)

    C_arr = np.array(C_arr)
    eps_arr = np.array(eps_arr)
    acc_arr = np.array(acc_arr)

    C_arr = np.unique(C_arr)
    eps_arr = np.unique(eps_arr)

    X, Y = np.meshgrid(eps_arr, C_arr)
    Z = acc_arr.reshape(len(C_arr), len(eps_arr))

    # Plot the 3D surface
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('C and epsilon Grid Search')
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('C')
    ax.set_zlabel('Percentage error')

    if save_C_eps_plot:
        plt.savefig(data_path, dpi=1200)

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
            filename = str(neigh) + "_SVR_test.png"
        else:
            filename = str(neigh) + "_SVR_train.png"

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/SVR_results/Models', neigh, filename)

        fig = plt.gcf()  # Get the current figure
        plt.savefig(data_path, dpi=150)
        print("[INFO]:", str(neigh), "plot predictions saved")
    
    plt.show()


def save_to_file(X_train, y_train, y_pred_train, X_test, y_test, y_pred_test, model_code):
    y_test = y_test.flatten()
    y_train = y_train.flatten()
    y_pred_test = y_pred_test.flatten()
    y_pred_train = y_pred_train.flatten()

    # Create dataframe with predicted and test values
    df_train = pd.DataFrame({'DWG_EXT': X_train[:, 1], 'ERF_EXT': X_train[:, 1], 'GAR': X_train[:, 12],
                             'NO BEDS': X_train[:, 2], 'CON': X_train[:, 4], 'SEC': X_train[:, 7], 'Predicted value': y_pred_train,
                            'Real value': y_train, 'Model code': model_code})

    df_test = pd.DataFrame({'DWG_EXT': X_test[:, 1], 'ERF_EXT': X_test[:, 1], 'GAR': X_test[:, 12],
                            'NO BEDS': X_test[:, 2], 'CON': X_test[:, 4], 'SEC': X_test[:, 7], 'Predicted value': y_pred_test,
                            'Real value': y_test, 'Error percentage': ((abs(y_pred_test-y_test))/y_test*100), 'Model code': model_code})

    # Write dataframes to file
    write_file_name = 'SVR_train_' + model_code + '.xlsx'
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/SVR_results', write_file_name)
    df_train.to_excel(data_path)

    write_file_name = 'SVR_test_' + model_code + '.xlsx'
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/SVR_results', write_file_name)
    df_test.to_excel(data_path)

    print("[INFO]: Files saved successfully")


def save_trained_model(model, degree, kernel, norm, stand, neigh):
    if norm:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_norm.sav"
    elif stand:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_stand.sav"
    else:
        filename = str(neigh) + "_" + str(kernel) + "_" + str(degree) + ".sav"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/SVR_results/Models', neigh, filename)

    # save the model to disk
    pickle.dump(model, open(data_path, 'wb'))


def save_predictions(y_test, y_pred_test, neigh, kernel, degree, stand, norm, test, save_fusion_pred, features):
    df = pd.DataFrame({'Predicted value': y_pred_test, 'Real value': y_test,
                      'Percentage error': ((abs(y_pred_test-y_test))/y_test*100)})

    df["DWEL_SIZE"] = features["DWEL_SIZE"]
    df["ERF_SIZE"] = features["ERF_SIZE"]

    if norm:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_norm.xlsx"
    elif stand:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_stand.xlsx"
    else:
        filename = str(neigh) + "_" + str(kernel) + "_" + str(degree) + ".xlsx"

    # Write dataframes to file
    if test:
        filename2 = str(neigh) + "_test_SVR.xlsx"
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/SVR_results/Test_predictions', filename)
        data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                      'Resources/Model_data/Fusion_results/Data/Test', neigh, filename2)
    else:
        filename2 = str(neigh) + "_SVR.xlsx"
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/SVR_results/Validation_predictions', filename)
        data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                      'Resources/Model_data/Fusion_results/Data/Train', neigh, filename2)

    if save_fusion_pred:
        df.to_excel(data_path2)

    df.to_excel(data_path)


def save_full_predictions(y_test, y_pred_test, X_test, neigh, stand, norm, kernel, degree, params):
    params.append("Predicted value")
    params.append("Real value")
    params.append("Percentage error")

    df_data = pd.DataFrame(X_test)
    df_targets = pd.DataFrame({'Predicted value': y_pred_test, 'Real value': y_test,
                               'Percentage error': ((abs(y_pred_test-y_test))/y_test*100)})

    df = pd.concat(
        [df_data, df_targets], axis=1)

    df.columns = params

    if norm:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_norm.xlsx"
    elif stand:
        filename = str(neigh) + "_" + str(kernel) + \
            "_" + str(degree) + "_stand.xlsx"
    else:
        filename = str(neigh) + "_" + str(kernel) + "_" + str(degree) + ".xlsx"

    # Write dataframes to file
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/SVR_results/Full_predictions', filename)

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


def main():
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
    opt_C_eps = False
    train_w_b = False # Only train/save the model with optimal parameter (Saves one sc_y)
    vali_para = False
    test = False

    # Save variables
    save_C_eps_plot = False
    save_model = False
    save_full_pred = False
    save_pred = False
    save_fusion_pred = False
    save_plt_pred = False

    # Plot variables
    plot_C_eps = False
    plt_pred = False
    metrics = False

    # Model parameters
    kernel = 'linear'  # linear, poly, rbf, sigmoid
    degree = 1
    gamma = 'auto'
    C = 0.1
    epsilon = 0.1
    shrinking = True

    features = import_features(neighbourhoods[neigh_num], test)

    # Training
    if opt_C_eps:
        print("[INFO]: Entered C and epsilon training")
        
        # Import data from a file
        neigh_df, Y_label = import_data(
            neighbourhoods[neigh_num], clean, norm, stand, test, opt_C_eps, params)
        
        eps_arr, C_arr, acc_arr, opt_C, opt_eps, opt_acc = train_C_and_epsilon(
            neigh_df, degree, kernel, gamma, shrinking, norm, stand, params)

        if plot_C_eps:
            plot_C_epsilon_opt(acc_arr, C_arr, eps_arr, save_C_eps_plot,
                               norm, stand, neighbourhoods[neigh_num], degree, opt_C, opt_eps, opt_acc)
        C = opt_C
        epsilon = opt_eps

        print("[INFO]: Optimal C and epsilon")
        print("       ", opt_C, "|", opt_eps, "|", opt_acc, "%")

        print("[INFO]: C and epsilon training complete")
    
    opt_C_eps = False
    
    # Import data from a file
    neigh_df, Y_label = import_data(
        neighbourhoods[neigh_num], clean, norm, stand, test, opt_C_eps, params)

    if train_w_b:
        print("[INFO]: Entered w and b training")
        SVR_regressor, sc_y = train_w_and_b(neigh_df, degree, kernel, gamma,
                                      shrinking, norm, stand, epsilon, C)

        if save_model:
            if norm:
                filename = str(neighbourhoods[neigh_num]) + "_" + str(kernel) + \
                    "_" + str(degree) + "_norm.sav"
            elif stand:
                filename = str(neighbourhoods[neigh_num]) + "_" + str(kernel) + \
                    "_" + str(degree) + "_stand.sav"
            else:
                filename = str(neighbourhoods[neigh_num]) + "_" + str(kernel) + "_" + str(degree) + ".sav"

            data_path = r''+os.path.join(os.path.dirname(__file__),
                                        'Resources/Model_data/SVR_results/Models', neighbourhoods[neigh_num], filename)
            data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                        'Resources/Model_data/SVR_results/Models', neighbourhoods[neigh_num], "scaler.pkl")

            # save the model to disk
            pickle.dump(SVR_regressor, open(data_path, 'wb'))
            pickle.dump(sc_y, open(data_path2, "wb"))          
            
            print("[INFO]: Saved trained model successfully")

        print("[INFO]: w and b training complete")

    # Validation
    if vali_para:
        print("[INFO]: Performing validation")
        SVR_regressor, sc_y = import_model(
            neighbourhoods[neigh_num], kernel, degree, norm, stand)

        # Populate X and y matrices
        _, X_test, _, y_test = populate_data_arrays(
            None, neigh_df, Y_label)

        # Computing model predictions
        y_pred_test = SVR_regressor.predict(X_test)

        # Scaling y_train
        if norm or stand:
            y_pred_test = y_pred_test.reshape(-1, 1)
            y_pred_test = sc_y.inverse_transform(y_pred_test)

        y_pred_test = y_pred_test.flatten()
        y_test = y_test.flatten()
        accuracy = perc_error(y_test, y_pred_test)
        print('[Accuracy]:', accuracy, '%')

        print("[INFO]: Validation complete")

        if save_pred:
            save_predictions(
                y_test, y_pred_test, neighbourhoods[neigh_num], kernel, degree, stand, norm, test, save_fusion_pred, features)

            print("[INFO]: Validation predictions saved successfully")

        if save_full_pred:
            save_full_predictions(
                y_test, y_pred_test, X_test, neighbourhoods[neigh_num], stand, norm, kernel, degree, params[:-1])

            print("[INFO]: Full validation predictions saved successfully")

        if plt_pred:
            plot_predictions(y_test, y_pred_test, neighbourhoods[neigh_num], save_plt_pred, test)
            
        if metrics:
            provide_metrics(y_test, y_pred_test)

    # Test
    if test:
        print("[INFO]: Performing test")
        SVR_regressor, sc_y = import_model(
            neighbourhoods[neigh_num], kernel, degree, norm, stand)

        # Populate X and y matrices
        _, X_test, _, y_test = populate_data_arrays(
            None, neigh_df, Y_label)

        # Computing model predictions
        y_pred_test = SVR_regressor.predict(X_test)

        if norm or stand:
            y_pred_test = y_pred_test.reshape(-1, 1)
            y_pred_test = sc_y.inverse_transform(y_pred_test)

        y_pred_test = y_pred_test.flatten()
        y_test = y_test.flatten()
        accuracy = perc_error(y_test, y_pred_test)
        print('[Accuracy]:', accuracy, '%')

        print("[INFO]: Test complete")

        if save_pred:
            save_predictions(
                y_test, y_pred_test, neighbourhoods[neigh_num], kernel, degree, stand, norm, test, save_fusion_pred, features)

            print("[INFO]: Test predictions saved successfully")

        if plt_pred:
            plot_predictions(y_test, y_pred_test, neighbourhoods[neigh_num], save_plt_pred, test)
            
        if metrics:
            provide_metrics(y_test, y_pred_test)

    print('[INFO]: Algorithm complete')

    completion_noise()


if __name__ == "__main__":
    main()
