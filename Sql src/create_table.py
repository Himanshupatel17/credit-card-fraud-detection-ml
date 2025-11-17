import pandas as pd
from sqlalchemy import create_engine

username="root"
password="Himanshu%4012"
servername="127.0.0.1"
database="fraud_db"
port="3306"

engine=create_engine(f"mysql+pymysql://{username}:{password}@{servername}:{port}/fraud_db")

transactions=pd.read_csv("D:\domain projects\credit card\dataset\Raw\creditcard_2023.csv")

transactions.to_sql("Transaction",con=engine,if_exists="replace",index=False)

print(f"tablecreated successfully ",database)