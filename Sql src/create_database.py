#import the libaries
import pandas as pd
from sqlalchemy import create_engine, text

# Database Connection 
username = "root"        
password = "Himanshu%4012"  
host = "127.0.0.1"       
port = "3306" 

#engiee for conn
engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}:{port}")

with engine.connect() as conn:
    conn.execute(text("CREATE DATABASE IF NOT EXISTS fraud_db;"))
    print("Database 'fraud_db' created successfully")