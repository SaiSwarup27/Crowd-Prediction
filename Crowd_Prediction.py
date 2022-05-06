import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings(action='ignore')

data=pd.read_csv('projects/data.csv')

def preprocess_inputs(df):
    df = df.copy()
    
    # Extract date features
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['hour'] = df['date'].apply(lambda x: x.hour)
    df['minute'] = df['date'].apply(lambda x: x.minute)
    df = df.drop('date', axis=1)
    # Split df into X and y
    y = df['number_people']
    X = df.drop('number_people', axis=1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test  = preprocess_inputs(data)

model=RandomForestRegressor()
model.fit(X_train, y_train)
def get_rmse(y_test, y_pred):
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))
    return rmse

def get_r2(y_test, y_pred):
    r2 = 1 - (np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2))
    return r2

y_pred = model.predict(X_test)
rmse = get_rmse(y_test, y_pred)
print(" RMSE: {:.2f}".format(rmse))

y_pred = model.predict(X_test)
r2 = get_r2(y_test, y_pred)
print(" R^2: {:.5f}".format(r2))

import tkinter as tk
from tkinter.constants import LEFT, RAISED, RIGHT, SUNKEN

window=tk.Tk()
window.title("Crowd Prediction")
frame1=tk.Frame(master=window,borderwidth=2,width=100,height=100,relief=SUNKEN)
frame1.pack()

label1=tk.Label(master=frame1,text="date: ")
label1.grid(row=0,column=0,sticky='e')
date=tk.Entry(master=frame1,relief=tk.SUNKEN,width=50)
date.grid(row=0,column=1)

label2=tk.Label(master=frame1,text="timestamp: ")
label2.grid(row=1,column=0,sticky='e')
timestamp=tk.Entry(master=frame1,relief=SUNKEN,width=50)
timestamp.grid(row=1,column=1)

label3=tk.Label(master=frame1,text="day_of_week: ")
label3.grid(row=2,column=0,sticky='e')
day_of_week=tk.Entry(master=frame1,relief=SUNKEN,width=50)
day_of_week.grid(row=2,column=1)

label4=tk.Label(master=frame1,text="is_weekend: ")
label4.grid(row=3,column=0,sticky='e')
is_weekend=tk.Entry(master=frame1,relief=SUNKEN,width=50)
is_weekend.grid(row=3,column=1)

label5=tk.Label(master=frame1,text="is_holiday: ")
label5.grid(row=4,column=0,sticky='e')
is_holiday=tk.Entry(master=frame1,relief=SUNKEN,width=50)
is_holiday.grid(row=4,column=1)

label6=tk.Label(master=frame1,text="temperature: ")
label6.grid(row=5,column=0,sticky='e')
temperature=tk.Entry(master=frame1,relief=SUNKEN,width=50)
temperature.grid(row=5,column=1)

label7=tk.Label(master=frame1,text="is_start_of_semister: ")
label7.grid(row=6,column=0,sticky='e')
is_start_of_semister=tk.Entry(master=frame1,relief=SUNKEN,width=50)
is_start_of_semister.grid(row=6,column=1)

label8=tk.Label(master=frame1,text="is_during_semister: ")
label8.grid(row=7,column=0,sticky='e')
is_during_semister=tk.Entry(master=frame1,relief=SUNKEN,width=50)
is_during_semister.grid(row=7,column=1)

label9=tk.Label(master=frame1,text="month: ")
label9.grid(row=8,column=0,sticky='e')
month=tk.Entry(master=frame1,relief=SUNKEN,width=50)
month.grid(row=8,column=1)

label10=tk.Label(master=frame1,text="hour: ")
label10.grid(row=9,column=0,sticky='e')
hour=tk.Entry(master=frame1,relief=SUNKEN,width=50)
hour.grid(row=9,column=1)

frame2=tk.Frame(master=window,borderwidth=2)
frame2.pack(fill=tk.X,ipadx=5,ipady=5)
button2=tk.Button(master=frame2,text='Submit',width=10,relief=RAISED,command=lambda: Output())
button2.grid(row=0,column=0,padx=20,ipadx=10)
entry1=tk.Entry(master=frame2,relief=RAISED)
entry1.grid(row=0,column=1,padx=10,ipadx=10)
entry1.insert(0,'No.of people: ')
entry2=tk.Entry(master=frame2,relief=RAISED,width=40)
entry2.grid(row=1,column=1)
entry2.insert(0,'Accuracy: ')
entry2.insert(10,(model.score(X_test, y_test)*100))

def Output():
    df=pd.DataFrame(columns=['date','timestamp','day_of_week','is_weekend','is_holiday','temperature','is_start_of_semister','is_during_semister','month','hour'])
    df.loc[0,'date']=date.get()
    df.loc[0,'timestamp']=timestamp.get()
    df.loc[0,'day_of_week']=day_of_week.get()
    df.loc[0,'is_weekend']=is_weekend.get()
    df.loc[0,'is_holiday']=is_holiday.get()
    df.loc[0,'temperature']=temperature.get()
    df.loc[0,'is_start_of_semister']=is_start_of_semister.get()
    df.loc[0,'is_during_semister']=is_during_semister.get()
    df.loc[0,'month']=month.get()
    df.loc[0,'hour']=hour.get()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['hour'] = df['date'].apply(lambda x: x.hour)
    df['minute'] = df['date'].apply(lambda x: x.minute)
    df = df.drop('date', axis=1)

    entry1.delete(14,tk.END)
    entry1.insert(14,model.predict(df))

    

window.mainloop()