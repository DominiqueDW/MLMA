from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from numpy import savetxt
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib
import winsound
import os.path
import pickle
import math


class MulPolynomialRegressor():
    # Constructor
    def __init__(self, degree=1, reg=False, lamda=0):
        self.coef = None
        self.degree = degree
        self.reg = reg
        self.lamda = lamda

    def generate_polynomial_matrix(self, X):
        degree = self.degree
        A = np.ones((X.shape[0], 1))

        for i in range(X.shape[1]):
            temp = (X[:, i][np.newaxis]).T
            A = np.append(A, temp, 1)

        if (degree > 1):
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    temp1 = (X[:, i][np.newaxis]).T
                    temp2 = (X[:, j][np.newaxis]).T
                    temp = np.multiply(temp1, temp2)
                    A = np.append(A, temp, 1)

        if (degree > 2):
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        temp1 = (X[:, i][np.newaxis]).T
                        temp2 = (X[:, j][np.newaxis]).T
                        temp3 = (X[:, k][np.newaxis]).T
                        temp = np.multiply(np.multiply(temp1, temp2), temp3)
                        A = np.append(A, temp, 1)

        if (degree > 3):
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        for l in range(k, X.shape[1]):
                            temp1 = (X[:, i][np.newaxis]).T
                            temp2 = (X[:, j][np.newaxis]).T
                            temp3 = (X[:, k][np.newaxis]).T
                            temp4 = (X[:, l][np.newaxis]).T
                            temp = np.multiply(np.multiply(
                                np.multiply(temp1, temp2), temp3), temp4)
                            A = np.append(A, temp, 1)

        return A

    def solve_for_coefficients(self, A, Y):
        if self.reg == False:
            A_inverse = np.linalg.pinv(A)
            self.coef = np.dot(A_inverse, Y)
        elif self.reg == True:
            A_transpose = A.T
            m = A.shape[1]
            I = np.identity(m)

            term1 = (1/m)*(np.dot(A_transpose, A)) + self.lamda*I
            term1_inverse = np.linalg.pinv(term1)
            term2 = (1/m)*np.dot(A_transpose, Y)

            self.coef = np.dot(term1_inverse, term2)
        else:
            print("[INFO]: reg is not of type boolean.")
            exit()

    def predict(self, X):
        if self.coef.all() == None:
            print(
                "[INFO]: Regression model coefficients do not exists. Unable to solve.")
        else:
            A_test = self.generate_polynomial_matrix(X)
            y_pred = np.dot(A_test, self.coef)

            return y_pred

    def print_coef(self):
        if self.coef is not None:
            print("[Coefficients]:", self.coef.shape)
            print(self.coef)
        else:
            print(
                "[INFO]: Regression model has not been fit. No coefficients to display.")

    def print_degree(self):
        if self.degree is not None:
            print("[Degree]:")
            print(self.degree)
        else:
            print(
                "[INFO]: Regression model has not been fit. No degree to display.")

    def print_lamda(self):
        if self.lamda is not None:
            print("[Lambda]:")
            print(self.lamda)
        else:
            print(
                "[INFO]: Regression model has not been fit. No lambda to display.")

    def save_coef(self, fold, degree, train_iter, norm, stand):
        if self.coef is not None:
            if norm:
                file_name = str(degree) + "_Coef_" + \
                    str(train_iter) + "_norm_" + str(fold)
            elif stand:
                file_name = str(degree) + "_Coef_" + \
                    str(train_iter) + "_stand_" + str(fold)
            else:
                str(degree) + "_Coef_" + str(train_iter) + "_" + str(fold)

            savetxt(file_name, self.coef, delimiter=",",
                    header="Coefficients for > 80% accuracy")
        else:
            print(
                "[INFO]: Regression model has not been fit. No coefficients to display.")

    def fit(self, X, Y):
        A = self.generate_polynomial_matrix(X)
        self.solve_for_coefficients(A, Y)

    def set_model(self, coefs):
        self.coef = coefs


def import_data(neigh, clean, norm, stand, test, opt_lam, params):
    if clean or opt_lam:
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


def import_model(neigh, degree, norm, stand):
    if norm:
        filename = str(neigh) + "_degree_" + str(degree) + "_norm.sav"
    elif stand:
        filename = str(neigh) + "_degree_" + str(degree) + "_stand.sav"
    else:
        filename = str(neigh) + "_degree_" + str(degree) + ".sav"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/MPR_results/Models', neigh, filename)

    MPR_regressor = pickle.load(open(data_path, 'rb'))

    return MPR_regressor


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


def binom(n, k):
    return math.comb(n, k)


def train_lambda(neigh_df, degree, plot_lambda, params, norm, stand):
    lamda = 0
    lam_iter = 500
    lam_incr = 0.1
    kfolds = 4
    
    lambda_arr = []
    acc_arr = []
    Y_label = 'SALE_PRICE_ESC'

    # Splitting dataset up into folds
    fold1, fold2, fold3, fold4 = kfold_cross_validation_random(
        neigh_df, params)

    for _ in tqdm(range(lam_iter), desc="Lambda iter"):
        lambda_arr.append(lamda)
        XV_acc_arr = []

        for i in range(kfolds):
            # Kfold cross validation
            df_train, df_test = split_train_test(
                fold1, fold2, fold3, fold4, iter=i)

            # Populate X and y matrices
            X_train, X_test, y_train, y_test = populate_data_arrays(
                df_train, df_test, Y_label)

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

            regressor = MulPolynomialRegressor(
                degree=degree, reg=True, lamda=lamda)
            regressor.fit(X_train, y_train)

            y_pred = regressor.predict(X_test)

            accuracy = perc_error(y_test, y_pred)

            XV_acc_arr.append(accuracy)

        acc_arr.append(np.mean(XV_acc_arr))

        lamda += lam_incr

    if plot_lambda:
        plot_lambda_acc(lambda_arr, acc_arr)

    ind = acc_arr.index(min(acc_arr))
    opt_acc = acc_arr[ind]
    opt_lambda = lambda_arr[ind]

    return opt_acc, opt_lambda


def train_coefs(neigh_df, degree, reg, lamda):
    Y_label = 'SALE_PRICE_ESC'

    # Populate X and y matrices
    _, X_test, _, y_test = populate_data_arrays(
        None, neigh_df, Y_label)

    # Creating and training and regressor model
    MPR_regressor = MulPolynomialRegressor(
        degree=degree, reg=reg, lamda=lamda)
    MPR_regressor.fit(X_test, y_test)

    print("[INFO]: Coefficient training complete")

    return MPR_regressor


def plot_lambda_acc(lamda_arr, acc_arr):
    max = acc_arr[np.argmax(acc_arr)]
    min = acc_arr[np.argmin(acc_arr)]

    plt.title("Lambda VS Accuracy")
    plt.plot(lamda_arr, acc_arr, color="b")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Lambda")
    plt.ylim(min-2, max+2)
    plt.xlim(0)
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
            filename = str(neigh) + "_MPR_test.png"
        else:
            filename = str(neigh) + "_MPR_train.png"

        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/MPR_results/Models', neigh, filename)

        fig = plt.gcf()  # Get the current figure
        plt.savefig(data_path, dpi=150)
        print("[INFO]:", str(neigh), "plot predictions saved")
    
    plt.show()


def save_to_file(X_train, y_train, y_pred_train, X_test, y_test, y_pred_test, accuracy, model_code):
    y_test = y_test.flatten()
    y_train = y_train.flatten()
    y_pred_test = y_pred_test.flatten()
    y_pred_train = y_pred_train.flatten()

    # Create dataframe with predicted and test values
    df_train = pd.DataFrame({'DWG_EXT': X_train[:, 1], 'ERF_EXT': X_train[:, 0], 'GAR': X_train[:, 12],
                             'NO BEDS': X_train[:, 2], 'CON': X_train[:, 4], 'SEC': X_train[:, 7], 'Predicted value': y_pred_train,
                            'Real value': y_train, 'Model code': model_code})

    df_test = pd.DataFrame({'DWG_EXT': X_test[:, 1], 'ERF_EXT': X_test[:, 0], 'GAR': X_test[:, 12],
                            'NO BEDS': X_test[:, 2], 'CON': X_test[:, 4], 'SEC': X_test[:, 7], 'Predicted value': y_pred_test,
                            'Real value': y_test, 'Error percentage': ((abs(y_pred_test-y_test))/y_test*100), 'Model code': model_code, 'Accuracy': accuracy[0]})

    # Write dataframes to file
    write_file_name = 'MPR_train_' + model_code + '.xlsx'
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/MPR_results', write_file_name)
    df_train.to_excel(data_path)
    write_file_name = 'MPR_test_' + model_code + '.xlsx'
    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/MPR_results', write_file_name)
    df_test.to_excel(data_path)

    print("[INFO]: Files saved successfully")


def save_trained_model(model, degree, norm, stand, neigh):
    if norm:
        filename = str(neigh) + "_degree_" + str(degree) + "_norm.sav"
    elif stand:
        filename = str(neigh) + "_degree_" + str(degree) + "_stand.sav"
    else:
        filename = str(neigh) + "_degree_" + str(degree) + ".sav"

    data_path = r''+os.path.join(os.path.dirname(__file__),
                                 'Resources/Model_data/MPR_results/Models', neigh, filename)

    # Save the model to disk
    pickle.dump(model, open(data_path, 'wb'))

    print("[INFO]: Saved trained model successfully")


def save_predictions(y_test, y_pred_test, neigh, degree, stand, norm, test, save_fusion_pred, features):
    df = pd.DataFrame({'Predicted value': y_pred_test, 'Real value': y_test,
                      'Percentage error': ((abs(y_pred_test-y_test))/y_test*100)})

    df["DWEL_SIZE"] = features["DWEL_SIZE"]
    df["ERF_SIZE"] = features["ERF_SIZE"]

    if norm:
        filename = str(neigh) + "_degree_" + str(degree) + "_norm.xlsx"
    elif stand:
        filename = str(neigh) + "_degree_" + str(degree) + "_stand.xlsx"
    else:
        filename = str(neigh) + "_degree_" + str(degree) + ".xlsx"

    # Write dataframes to file
    if test:
        filename2 = str(neigh) + "_test_MPR.xlsx"
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/MPR_results/Test_predictions', filename)
        data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                      'Resources/Model_data/Fusion_results/Data/Test', neigh, filename2)
    else:
        filename2 = str(neigh) + "_MPR.xlsx"
        data_path = r''+os.path.join(os.path.dirname(__file__),
                                     'Resources/Model_data/MPR_results/Validation_predictions', filename)
        data_path2 = r''+os.path.join(os.path.dirname(__file__),
                                      'Resources/Model_data/Fusion_results/Data/Train', neigh, filename2)

    if save_fusion_pred:
        df.to_excel(data_path2)

    df.to_excel(data_path)


def weighted_average(values, weightings):
    weights = np.zeros(len(values))
    acc_sum = sum(weightings)
    w_avg = 0

    for i in range(len(values)):
        weights[i] = weightings[i]/acc_sum

    for i in range(len(values)):
        w_avg += values[i]*weights[i]

    return w_avg


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


def main():  # Closed-form solution
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
    opt_lam = False
    training_coef = False
    vali_para = False
    test = False

    # Save variables
    save_model = False
    save_pred = False
    save_fusion_pred = False
    save_plt_pred = False

    # Plotting variables
    plot_lambda = False
    plt_pred = False
    show_coef = False
    show_model = False
    metrics = False

    # Model parameters
    degree = 1
    reg = 1
    lamda = 0

    features = import_features(neighbourhoods[neigh_num], test)

    # Training
    if opt_lam:
        print("[INFO]: Entered lambda training")

        # Import data from a file
        neigh_df, Y_label = import_data(
            neighbourhoods[neigh_num], clean, norm, stand, test, opt_lam, params)

        # Optimize lambda
        opt_acc, opt_lambda = train_lambda(
            neigh_df, degree, plot_lambda, params, norm, stand)
        print("[INFO]: Optimal lambda for deg", degree, ":")
        print("       ", opt_lambda, " | ", (100-opt_acc), "%")

        lamda = round(opt_lambda, 2)

        print("[INFO]: Lambda training complete")

    opt_lambda = False

    # Import data from a file
    neigh_df, Y_label = import_data(
        neighbourhoods[neigh_num], clean, norm, stand, test, opt_lam, params)

    if training_coef:
        print("[INFO]: Entered coefficient training")

        MPR_regressor = train_coefs(neigh_df, degree, reg, lamda)

        if save_model:
            save_trained_model(MPR_regressor, degree, norm,
                               stand, neighbourhoods[neigh_num])

    # Validation
    if vali_para:
        print("[INFO]: Performing validation")
        MPR_regressor = import_model(
            neighbourhoods[neigh_num], degree, norm, stand)

        # Populate X and y matrices
        _, X_test, _, y_test = populate_data_arrays(
            None, neigh_df, Y_label)

        # Computing model predictions
        y_pred_test = MPR_regressor.predict(X_test)
        accuracy = perc_error(y_test, y_pred_test)

        print('[Accuracy]:', accuracy, '%')

        print("[INFO]: Validation complete")

        y_pred_test = y_pred_test.flatten()
        y_test = y_test.flatten()
        if save_pred:
            save_predictions(
                y_test, y_pred_test, neighbourhoods[neigh_num], degree, stand, norm, test, save_fusion_pred, features)

            print("[INFO]: Validation predictions saved successfully")

        if plt_pred:
            plot_predictions(y_test, y_pred_test, neighbourhoods[neigh_num], save_plt_pred, test)

        if show_coef:
            MPR_regressor.print_coef()
            
        if metrics:
            provide_metrics(y_test, y_pred_test)

    # Test
    if test:
        print("[INFO]: Performing test")
        _, X_test, _, y_test = populate_data_arrays(
            None, neigh_df, Y_label)

        MPR_regressor = import_model(
            neighbourhoods[neigh_num], degree, norm, stand)

        # Predicting data point values
        y_pred_test = MPR_regressor.predict(X_test)
        accuracy = perc_error(y_test, y_pred_test)

        print("[Accuracy]: ", accuracy, "%")

        print("[INFO]: Test complete")

        y_pred_test = y_pred_test.flatten()
        y_test = y_test.flatten()
        
        if save_pred:
            save_predictions(
                y_test, y_pred_test, neighbourhoods[neigh_num], degree, stand, norm, test, save_fusion_pred, features)

            print("[INFO]: Test predictions saved successfully")

        if plt_pred:
            plot_predictions(y_test, y_pred_test, neighbourhoods[neigh_num], save_plt_pred, test)
            
        if metrics:
            provide_metrics(y_test, y_pred_test)

    # Model properties
    if show_model:
        MPR_regressor = import_model(
            neighbourhoods[neigh_num], degree, norm, stand)

        MPR_regressor.print_coef()
        MPR_regressor.print_degree()
        MPR_regressor.print_lamda()

    print("[INFO]: Algorithm complete")

    completion_noise()


if __name__ == "__main__":
    main()
