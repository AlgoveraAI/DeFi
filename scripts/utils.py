import pandas as pd

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