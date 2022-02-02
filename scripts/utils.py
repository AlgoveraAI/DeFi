import pandas as pd
from fastai.tabular.all import *

def get_token_features(df, tokens):
    df1 = pd.DataFrame()
    for tok in tokens:
        df_tok = df[df['Token']==tok]
        df_tok = df_tok.drop(['Token', 'Date'], axis=1)

        col_names = []
        for col in df_tok.columns:
            if col == 'Timestamp':
                col_names.append(f'{col}')
            else:
                col_names.append(f'{tok}_{col}')
            
        df_tok.columns = col_names
        #df_tok = df_tok.set_index('Timestamp', drop=True)
        
        if df1.empty:
            df1 = df_tok
        else:
            df1 = pd.merge(df1, df_tok, on='Timestamp')


def get_target(row, target_column, target_window):

    try:
        target = df1[df1['Timestamp'] == row['Timestamp'] + 1800.0*target_window][target_column].values[0]
    except:
        target = np.NaN
    
    return target

def get_tabpandas_singletimestep(df, tokens, target_window):

    y_names = []
    for tok in tokens:
        target = f'{tok}_Target'
        y_names.append(target)
        target_column = f'{tok}_Borrowing Rate'
        df[target] =  df.apply(lambda x: get_target(x, target_column, target_window), axis=1)

    df = df.dropna()
    df = df.drop(['Timestamp', 'Date'], axis=1)
    
    df['Train'] = None
    train_index = int(len(df)*0.8)
    df.loc[:train_index, 'Train'] = True
    df.loc[train_index:, 'Train'] = False
    
    df = df.reset_index(drop=True)
    splits = (list(df[df['Train']==True].index), list(df[df['Train']==False].index))
    
    df = df.drop(['Train'], axis=1)

    cont_names = list(df.columns[:len(tokens)])

    procs = [Categorify, FillMissing, Normalize]
    y_block = RegressionBlock()

    to = TabularPandas(df, procs=procs, cont_names=cont_names,
                       y_names=y_names, y_block=y_block, splits=splits)
    dls = to.dataloaders(bs=128)

    return to, dls

def get_tabpandas_multi(df, target_window, n_timepoint_inp):

    df = df.reset_index(drop=True)
    feature_cols = ['DAI_Borrowing Rate', 'DAI_Deposit Rate', 'DAI_Borrow Volume', 'DAI_Supply Volume', 
                    'USDC_Borrowing Rate', 'USDC_Deposit Rate', 'USDC_Borrow Volume', 'USDC_Supply Volume', 
                    'USDT_Borrowing Rate', 'USDT_Deposit Rate', 'USDT_Borrow Volume', 'USDT_Supply Volume',
                    'ETH_Borrowing Rate', 'ETH_Deposit Rate', 'ETH_Borrow Volume', 'ETH_Supply Volume']

    target_columns = ['DAI_Borrowing Rate', 'USDC_Borrowing Rate', 'USDT_Borrowing Rate', 'ETH_Borrowing Rate']

    cols_names = []
    for j in range(n_timepoint_inp):
        for col in feature_cols:
            cols_names.append(f'{col}_t-{n_timepoint_inp -j-1}')
    cols_names += target_columns

    pairs = []
    for i, row in tqdm(df.iterrows()):
        if i < (len(df)-target_window-n_timepoint_inp-1):
            features = df.loc[i:i+n_timepoint_inp-1, feature_cols].values
            features = [item for sublist in features for item in sublist]
            targ = list(df.loc[i+n_timepoint_inp-1+target_window, target_columns].values)
            features += targ
            pairs.append(features)

    df = pd.DataFrame(pairs, columns=cols_names)
    df = df.dropna()
    df = df.reset_index(drop=True)

    #train_test_split
    df['Train'] = None
    train_index = int(len(df)*0.8)
    df.loc[:train_index, 'Train'] = True
    df.loc[train_index:, 'Train'] = False

    splits = (list(df[df['Train']==True].index), list(df[df['Train']==False].index))

    df = df.drop(['Train'], axis=1)

    cont_names = list(df.columns[:-4])

    procs = [Categorify, FillMissing, Normalize]
    y_block = RegressionBlock()

    to = TabularPandas(df, procs=procs, cont_names=cont_names, y_names=target_columns, y_block=y_block, splits=splits)
    dls = to.dataloaders(bs=128)

    return to, dls