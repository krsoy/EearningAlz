import pickle
import pandas as pd
# ./data/motley-fool-data.pkl
data = pickle.loads(open("../data/motley-fool-data.pkl", "rb").read())
print(data)
# raw time formate
# Nov 18, 2021, 12:00 p.m. ET

# convert date column into datetime
cleaned = (
    data['date']
    .str.replace(r'\s+ET$', '', regex=True)          # strip trailing " ET"
    .str.replace(r'\.(?=\s)', '', regex=True)         # remove "Aug." -> "Aug"
    .str.replace('a.m.', 'AM', regex=False)
    .str.replace('p.m.', 'PM', regex=False)
)

data['date'] = pd.to_datetime(cleaned, format='mixed')
print(data['date'])