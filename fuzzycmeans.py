import numpy as np
import pandas as pd
import skfuzzy as fuzzy
from datetime import date
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



# Taken from
# https://2-bitbio.com/post/clustering-rnaseq-data-using-fuzzy-c-means-clustering/
def m_estimate(df):
  N =  df.shape[0]
  D = df.shape[1]
  m = 1 + (1418/N + 22.05)*D**(-2) + (12.33/N +0.243)*D**(-0.0406*np.log(N) - 0.1134)
  return(m)


def apply_fcm_test(n_clusters_list, models, reduced_df_np_test):

    pc_list = []
    pec_list = []
    fcm_soft_labels_list = []
    num_clusters = len(n_clusters_list)
    rows = int(np.ceil(np.sqrt(num_clusters)))
    cols = int(np.ceil(num_clusters / rows))

    if reduced_df_np_test.shape[1] == 2:
        fig, axes = plt.subplots(rows, cols, figsize=(20,24))

        for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
            pc = model.partition_coefficient
            pec = model.partition_entropy_coefficient
            pc_list.append(pc)
            pec_list.append(pec)
            fcm_centers = model.centers
            fcm_labels = model.predict(reduced_df_np_test)
            fcm_soft_labels = model.soft_predict(reduced_df_np_test)
            fcm_soft_labels_list.append(fcm_soft_labels)
            axe.scatter(reduced_df_np_test[:,0], reduced_df_np_test[:,1], c=fcm_labels, alpha=.4, s=25)
            axe.set_xlabel('c1')
            axe.set_ylabel('c2')
            axe.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=200, c='red')
            axe.set_title(f'n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}')

    elif reduced_df_np_test.shape[1] == 3:
        fig = plt.figure(figsize=(20,24))
        for n_clusters, model in zip(n_clusters_list, models):
            idx = n_clusters - 1
            ax = fig.add_subplot(rows, cols, idx, projection='3d')
            pc = model.partition_coefficient
            pec = model.partition_entropy_coefficient
            pc_list.append(pc)
            pec_list.append(pec)
            fcm_centers = model.centers
            fcm_labels = model.predict(reduced_df_np_test)
            fcm_soft_labels = model.soft_predict(reduced_df_np_test)
            fcm_soft_labels_list.append(fcm_soft_labels)
            ax.scatter(reduced_df_np_test[:,0], reduced_df_np_test[:,1], reduced_df_np_test[:,2], c=fcm_labels, alpha=.4, s=25)
            ax.set_xlabel('c1')
            ax.set_ylabel('c2')
            ax.set_zlabel('c3')
            ax.scatter(fcm_centers[:,0], fcm_centers[:,1], fcm_centers[:,2], marker="+", s=200, c='red', zorder=1)
            ax.set_title(f'n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}')

    else:
        for n_clusters, model in zip(n_clusters_list, models):
            idx = n_clusters - 1
            pc = model.partition_coefficient
            pec = model.partition_entropy_coefficient
            pc_list.append(pc)
            pec_list.append(pec)
            fcm_centers = model.centers
            fcm_labels = model.predict(reduced_df_np_test)
            fcm_soft_labels = model.soft_predict(reduced_df_np_test)
            fcm_soft_labels_list.append(fcm_soft_labels)
            print(f'n_clusters = {n_clusters}, PC = {pc:.3f}, PEC = {pec:.3f}')
        return pc_list, pec_list, fcm_centers, fcm_labels, fcm_soft_labels_list

    fig.tight_layout(pad=1.5)
    plt.show()
    return pc_list, pec_list, fcm_centers, fcm_labels, fcm_soft_labels_list


def create_columns_list(number_components):
    cols = []
    for i in range(number_components):
        col = "c" + str(i)
        cols.append(col)
    return cols


def apply_pca(df, number_components=3, pca_fitted=None):
    cols_list = create_columns_list(number_components)
    if pca_fitted != None:
        print('Fitting data using provided PCA object.')
        pca_fitted.transform(df)
        reduced_df = pd.DataFrame(pca_fitted.transform(df), columns=(cols_list))
        print(reduced_df.describe().T)
        return reduced_df, pca_fitted
    pca = PCA(n_components=number_components)
    pca.fit(df)
    reduced_df = pd.DataFrame(pca.transform(df), columns=(cols_list))
    print(reduced_df.describe().T)
    return reduced_df, pca


def get_dataframe(csv_path):
    df = pd.read_csv(csv_path, sep='\t', lineterminator='\n')
    print('> Found dataframe with', str(len(df.index)), 'lines.')
    return df


def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def get_educational_group(educational_str):
    if educational_str == 'Basic':
        return 0
    if educational_str == '2n Cycle':
        return 1
    if educational_str == 'Graduation':
        return 2
    if educational_str == 'Master':
        return 3
    if educational_str == 'PhD':
        return 4


def get_marital_group(marital_status):
    if marital_status in ['Alone', 'Absurd', 'YOLO', 'Single']:
        return 0
    if marital_status == 'Widow':
        return 1
    if marital_status == 'Divorced':
        return 2
    if marital_status == 'Together':
        return 3
    if marital_status == 'Married':
        return 4


def get_sharing_group(marital_status):
    if marital_status in ['Alone', 'Absurd', 'YOLO', 'Single', 'Divorced', 'Widow']:
        return 0
    else:
        return 1


def get_age(year_birth, current_year):
    ano = int(year_birth)
    age = current_year - ano
    return age


def get_customer_time(dt_costumer, most_recent_date):
    return (most_recent_date - dt_costumer).days


def normalize_z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std

