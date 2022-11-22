from sklearn.model_selection import train_test_split
import pandas as pd


csv_file = './datasets/MNIST/raw/mnist_train.csv'
df = pd.read_csv(csv_file, header=None)
df = df.sample(frac=1).reset_index(drop=True)

X_train, X_val, _, _ = train_test_split(df, df.iloc[:, :1], train_size=0.9)

X_train.to_csv('./datasets/MNIST/raw/mnist_train.csv', index=False, header=None)
X_val.to_csv('./datasets/MNIST/raw/mnist_val.csv', index=False, header=None)
