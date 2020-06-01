import os
import pandas as pd
import numpy as np
import math
import re
import matplotlib.pyplot as plt


def data_load(DATASET_DIR, DATA_TXT):
    # Loading Data Files
    df = pd.read_csv(
                    os.path.join(DATASET_DIR, DATA_TXT),
                    header=None,
                    names=['movie_id', 'rating', 'date'],
                    usecols=[0,1,2]
                    )
    return df


                
def data_save(df, DATA_file):
    print('\nStarting save to .csv file')

    df.to_csv(
            DATA_file,
            sep=',',
            header=True,
            index=False,
            columns=df.columns #['customer_id', 'movie_id', 'rating', 'date', 'customer_emb_id', 'movie_emb_id']
            )
    print('Saved to ', DATA_file)

    

def add_customer_id(df_1, customer_id_1):
    for idx in range(len(customer_id_1)-1):
        df_1.loc[customer_id_1[idx]:customer_id_1[idx+1]-1, 'customer_id'] = df_1.loc[customer_id_1[idx], 'movie_id'][:-1]
    df_1.loc[customer_id_1[-1]:, 'customer_id'] = df_1.loc[customer_id_1[-1], 'movie_id'][:-1]
    df_1.dropna(inplace=True)
    df_1 = df_1.reset_index(drop=True)


    '''for idx in range(len(customer_id_2)-1):
        df_2.loc[customer_id_2[idx]:customer_id_2[idx+1]-1, 'customer_id'] = df_2.loc[customer_id_2[idx], 'movie_id'][:-1]
    df_2.loc[customer_id_2[-1]:, 'customer_id'] = df_2.loc[customer_id_2[-1], 'movie_id'][:-1]
    df_2.dropna(inplace=True)
    df_2 = df_2.reset_index(drop=True)'''
    return df_1



def concatenate_plot_data(df_1, df_2):
    df_1['rating'] = df_1['rating'].astype('float')
    df_2['rating'] = df_2['rating'].astype('float')

    # Print useful info about the Data
    print('Dataset 1 shape: {}'.format(df_1.shape))
    print('Dataset 2 shape: {}'.format(df_2.shape))

    # concatenate the Data
    df = pd.concat([df_1, df_2]).reset_index(drop=True)

    print('Full Dataset shape: {}'.format(df.shape))
    print('\n --- Complete Dataset example ---')
    print(df.iloc[::1000, :],'\n')

    # Data Viewing
    rating_distribution = df.groupby('rating')['rating'].agg(['count'])

    movie_count = len(df['movie_id'].unique())

    customer_count = len(df['customer_id'].unique())
    #customer_count = df['customer_id'].nunique() - movie_count

    rating_count = len(df[df.rating.isnull()==False]['rating'])
    #rating_count = df['customer_id'].count() - movie_count


    ax = rating_distribution.plot(kind = 'barh', legend = False, figsize = (15,10))
    plt.title('Total pool: {:,} Movies, {:,} Customers, {:,} Ratings given'.format(movie_count, customer_count, rating_count), fontsize=20)
    plt.axis('off')

    for i in range(1,6):
        ax.text(rating_distribution.iloc[i-1][0]/4, i-1, 'Rating {}: {:.0f}%'.format(i, rating_distribution.iloc[i-1][0]*100 / rating_distribution.sum()[0]), color = 'white', weight = 'bold')

    plt.show()

    return df



def filter_data(df):
    # remove those Movie ID rows
    df = df[pd.notnull(df['rating'])].reset_index(drop=True)
    df = df[['customer_id', 'movie_id', 'rating', 'date']]

    df['movie_id'] = df['movie_id'].astype(int)
    df['customer_id'] = df['customer_id'].astype(int)

    print('\n --- New Dataset examples ---')
    print(df.iloc[::1000, :])


    # Delete the Outliers -> Better Performance
    f = ['count','mean']

    df_movie_summary = df.groupby('movie_id')['rating'].agg(f)
    df_movie_summary.index = df_movie_summary.index.map(int)
    movie_benchmark = round(df_movie_summary['count'].quantile(0.8), 0)
    drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index
    print(df_movie_summary)

    df_cust_summary = df.groupby('customer_id')['rating'].agg(f)
    df_cust_summary.index = df_cust_summary.index.map(int)
    df_cust_summary = df_cust_summary[df_cust_summary['count'] <= 2000]
    customer_benchmark = round(df_cust_summary['count'].quantile(0.8), 0)
    drop_customer_list = df_cust_summary[df_cust_summary['count'] < customer_benchmark].index
    print(df_cust_summary)

    print('Movie minimum times of review: {}'.format(movie_benchmark))
    print('Customer minimum times of review: {}'.format(customer_benchmark))

    print('Original Shape: {}'.format(df.shape))
    df = df[~df['movie_id'].isin(drop_movie_list)].reset_index(drop=True)
    df = df[~df['customer_id'].isin(drop_customer_list)].reset_index(drop=True)
    print('After Trim Shape: {}'.format(df.shape))

    df['customer_emb_id'] = df['customer_id'] - 1
    df['movie_emb_id'] = df['movie_id'] - 1

    print('\n --- After Trim Data Examples ---')
    print(df.iloc[::10000, :])
    return df, df_movie_summary, movie_benchmark, df_cust_summary, customer_benchmark