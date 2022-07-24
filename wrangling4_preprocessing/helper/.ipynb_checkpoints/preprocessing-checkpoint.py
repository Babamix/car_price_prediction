import numpy as np
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import normalize

class CategoricalFeatures:
    def __init__(self, df, state, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas dataframe
        categorical_features: list of categorical column names e.g. nominal, ordinal data type
        encoding_type: type of encoding e.g. label, one_hot
        handle_na: handle the missing values or not e.g. True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type  = encoding_type
        self.state = state
        self.handle_na = handle_na

        if self.handle_na is True:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        if self.state == 'test':
            with open('artifacts/label_encoders.pkl' , 'rb') as f:
                label_encoders = pickle.load(f)
        else:
            label_encoders = preprocessing.LabelEncoder()
            label_encoders.fit(self.df[self.cat_feats].values)
        self.output_df[self.cat_feats] = label_encoders.transform(self.df[self.cat_feats].values)
        return self.output_df

    def _one_hot_encoding(self):
        print("Halo")
        if self.state == 'test':
            with open('artifacts/one_hot_encoders.pkl' , 'rb') as f:
                one_hot_encoders = pickle.load(f)
        else:
            one_hot_encoders = preprocessing.OneHotEncoder()
            one_hot_encoders.fit(self.df[self.cat_feats].values)
            pickle.dump(one_hot_encoders, open("artifacts/one_hot_encoders.pkl", "wb"))
        dum_ct = pd.DataFrame(one_hot_encoders.transform(self.df[self.cat_feats].values).toarray(), index = self.df.index)
        self.output_df = self.df.drop(columns=self.cat_feats, axis=1).join(dum_ct) 
        return self.output_df                        

    def _get_dummies(self):
        self.output_df = pd.get_dummies(self.df, columns=self.cat_feats, dummy_na=False)
        return self.output_df

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "one_hot":   
            return self._one_hot_encoding()
        elif self.enc_type == "get_dum":
            return self._get_dummies()
        else:
            raise Exception("Encoding type not supported!")
            

def standard_scaler(df: pd.DataFrame):
    """Scaling standard scaler transform."""
    index_cols = df.index
    scaler = preprocessing.StandardScaler()
    np_scaler = scaler.fit_transform(df)
    df_transformed = pd.DataFrame(
        np_scaler, index=index_cols, columns=df.columns
    )
    return scaler, df_transformed