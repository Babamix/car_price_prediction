import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import pickle 

def feature_engineering(df, state):
    
    df.rename(columns = {"dateCreated": "ad_created",
                                "dateCrawled": "date_crawled",
                                "fuelType": "fuel_type",
                                "lastSeen": "last_seen",
                                "monthOfRegistration": "registration_month",
                                "notRepairedDamage": "unrepaired_damage",
                                "nrOfPictures": "num_of_pictures",
                                "offerType": "offer_type",
                                "postalCode": "postal_code",
                                "powerPS": "power_ps",
                                "vehicleType": "vehicle_type",
                                "yearOfRegistration": "registration_year"}, inplace = True)
    
    # ubah format ke datetime
    df[["ad_created", "date_crawled", "last_seen"]] = df[["ad_created", "date_crawled",
                                                                        "last_seen"]].apply(pd.to_datetime)
    
    # membersihkan simbol tidak penting
    df['price'] = df['price'].str.replace(',', '').apply(lambda x: x.split('$')[1])
    df['odometer'] = df['odometer'].str.replace(',', '').apply(lambda x: x.split('km')[0])
    df[["price", "odometer"]] = df[["price", "odometer"]].apply(pd.to_numeric)
    
    #drop kolom yang perbandingannya terlalu besar (kolom seller & offer_type)
    #drop kolom num_of_pictures karena tidak berisi sama sekali
    #drop kolom name dan postal_code
    df.drop(columns=['seller', 'offer_type', 'num_of_pictures', 'name', 'postal_code'], inplace=True)
    
    # menghilangkan outlier
    df = df[(df['price'] >= 500) & (df['price'] <= 40000)]
    
    # imputasi kolom NaN
    for column in df.columns:
        if df[column].dtype == 'O':
            df[column].fillna(df[column].mode()[0], inplace=True)
        elif df[column].dtype == 'int64':
            df[column].fillna(df[column].median(), inplace=True)
    
    # normalisasi data numerik
    numeric_columns = []
    for column in df.columns:
        if (df[column].dtype == 'int64') & (column != 'price'):
            numeric_columns.append(column)
        
    df[numeric_columns] = normalize(df[numeric_columns])
    
    # encode 
    df['unrepaired_damage'] = df['unrepaired_damage'].apply(lambda x: 1 if x == "ja" else 0)
    df['gearbox'] = df['gearbox'].apply(lambda x: 1 if x == "manuell" else 0)
    df['abtest'] = df['abtest'].apply(lambda x: 1 if x == "test" else 0)
    df = pd.get_dummies(df, columns=['vehicle_type','fuel_type','brand'])
    df.drop(['model'], axis=1, inplace=True)
    
    df_transformed = df
    return df_transformed