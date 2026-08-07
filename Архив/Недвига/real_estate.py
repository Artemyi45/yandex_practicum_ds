import pandas as pd

pd.set_option('display.max_columns', 30)
data = pd.read_csv('/Users/test/Desktop/Яндекс Практикум/Data Sets/real_estate_data.csv', sep='\t')

data['last_price'] = data['last_price'].replace(12190.0, 12190000.0) / 1000000

data['first_day_exposition'] = pd.to_datetime(data['first_day_exposition'], format='%Y-%m-%dT%H:%M:%S')

data.loc[data['studio'], 'rooms'] = 1
mask = (data['total_area'] <= 45) & (data['rooms'] == 0)
data.loc[mask, 'rooms'] = 1
data['rooms'] = data['rooms'].replace(0, 7)

data['ceiling_height'] = data['ceiling_height'].replace({24.0: 2.40,
                                                         25.0: 2.50,
                                                         26.0: 2.60,
                                                         27.0: 2.70,
                                                         27.5: 2.75,
                                                         32.0: 3.20,
                                                         100.0: 10.0,
                                                         1.00: 2.50,
                                                         1.20: 2.50,
                                                         1.75: 2.50,
                                                         2.00: 2.50,
                                                         14.0: 2.50,
                                                         20.0: 2.50,
                                                         22.6: 2.26})
potolki = data.groupby('total_area')['ceiling_height'].mean()
data['ceiling_height_new'] = data['total_area'].map(lambda x: potolki[x])
data['ceiling_height'] = data['ceiling_height'].fillna(value=data['ceiling_height_new'])
data['ceiling_height'] = round(data['ceiling_height'], 2)
del data['ceiling_height_new']

missing_living = data[data['living_area'].isnull()]
data.loc[missing_living.index, 'living_area'] = data.loc[missing_living.index, 'total_area'].multiply(0.56)

missing_kitchen = data[data['kitchen_area'].isnull()]
data.loc[missing_kitchen.index, 'kitchen_area'] = data.loc[missing_kitchen.index, 'total_area'].multiply(0.18)

data['balcony'] = data['balcony'].fillna(0)

grouped = data.groupby('locality_name')['days_exposition'].mean()
data['days_exposition'] = grouped['days_exposition'].transform(lambda x: x.fillna(x.mean()))