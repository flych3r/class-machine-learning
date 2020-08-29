def min_max_scaler(df):
    return (df - df.min()) / (df.max() - df.min())

def standard_scaler(df):
    return (df - df.mean()) / df.std()

def split_data(df, test_size=0.2):
    df = df.sample(frac=1)
    size = int(len(df) * test_size)
    test = df.iloc[:size]
    train = df.iloc[size:]
    return train, test
