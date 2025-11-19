# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 10:28:40 2025

@author: Romain

source : https://github.com/building-energy/epw
"""

from epw import epw
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


'Reading a .epw file:'
a=epw()
a.read(r'C:\Users\Corentin\TFEtest\virtualenv\datasetweather\Brussels.Natl.AP_BEL.epw')
print(a)

'Viewing the header information:'
d=a.headers # this is a dictionary of the header information
#print(d)

'Viewing the climate data'
df=a.dataframe  # this is pandas dataframe

df1=df[['Year', 'Month', 'Day', 'Hour', 'Minute','Dry Bulb Temperature','Global Horizontal Radiation','Liquid Precipitation Depth']]
#print(df1.head())
df1['datetime'] = pd.to_datetime(
    dict(
        year=2025,              # or df['year'] if you also have a year column/ leave it as is since it's TMY with different years
        month=df['Month'],
        day=df['Day'], 
    )
)

daily = df.groupby(df1['datetime']).agg({
    'Dry Bulb Temperature': 'mean',        # daily mean temperature
    'Liquid Precipitation Depth': 'sum',       # daily total precipitation
    'Global Horizontal Radiation': 'sum'                  # daily total radiation
})

daily['Global Horizontal Radiation'] = daily['Global Horizontal Radiation']/1000; #en kW
#daily.to_excel("daily_data.xlsx", index ='datetime')
#df.to_excel("epw_Brussels_airport.xlsx")


fig, axes = plt.subplots(3, 1, figsize=(14, 10))  # width=14, height=10

plt.subplot(3,1,1)
plt.plot(daily.index, daily['Dry Bulb Temperature'])
plt.title("Température journalière moyenne")
plt.grid()
plt.ylabel('Température [°C]')

plt.subplot(3,1,2)
plt.plot(daily.index, daily['Global Horizontal Radiation'])
plt.title("Ensoleillement total journalier")
plt.grid()
plt.ylabel('ensoleillement [kWh/$m^{2}$]')

plt.subplot(3,1,3)
plt.plot(daily.index, daily['Liquid Precipitation Depth'])
plt.title("Précipitation totale journalière")
plt.ylabel('Précipitation [mm]')
plt.xlabel('Date')
plt.grid()

plt.tight_layout()
plt.show()

