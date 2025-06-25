import pandas as pd
import config
from sklearn.model_selection import train_test_split

def process():
    # 读取数据
    df = pd.read_csv(config.RAW_DATA_DIR / 'cmn.txt',header=None,sep='\t',usecols=[0,1],names=['en', 'zh'],encoding='utf-8')

    # 过滤空值
    df = df.dropna()
    df = df[df['en'].str.strip().ne('') & df['zh'].str.strip().ne('')]

    # 划分数据集
    train_df, test_df = train_test_split(df, test_size=0.2)

    # 构建词表




if __name__ == '__main__':
    process()