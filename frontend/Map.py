import pandas as pd
import streamlit as st


from streamlit_folium import st_folium
import folium
from folium.plugins import AntPath

from pathlib import Path

# load dummy dataset
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "dummy_dataset.csv"

df = pd.read_csv(file_path, keep_default_na=False)

st.title("Map")
#Create Map
m= folium.Map(location=[30,-120], zoom_start=1)


countries = sorted(df["country"].unique()) # list of countries

top5 = (
    df.groupby("country")["trade_value"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index
    .tolist()
)





df.set_index('country',inplace=True)

#set direction of the ant paths
startLA=df.at['China','latitude'][1]
startLO=df.at['China','longitude'][1]
endLA=df.at[top5[0],'latitude'][1]
endLO=df.at[top5[0],'longitude'][1]

endLA2=df.at[top5[1],'latitude'][1]
endLO2=df.at[top5[1],'longitude'][1]

endLA3=df.at[top5[2],'latitude'][1]
endLO3=df.at[top5[2],'longitude'][1]

endLA4=df.at[top5[3],'latitude'][1]
endLO4=df.at[top5[3],'longitude'][1]


endLA5=df.at[top5[4],'latitude'][1]
endLO5=df.at[top5[4],'longitude'][1]

print(startLA)



path1= [(startLA,startLO),(endLA,endLO)]
path2= [(startLA,startLO),(endLA2,endLO2)]
path3= [(startLA,startLO),(endLA3,endLO3)]
path4= [(startLA,startLO),(endLA4,endLO4)]
path5= [(startLA,startLO),(endLA5,endLO5)]

name1=top5[0]
name2=top5[1]
name3=top5[2]
name4=top5[3]
name5=top5[4]





def Arrow(x,y):
    AntPath(x,delay=100,weight=3, color="white",pulse_color="green",dash_array=[30,15],tooltip=y).add_to(m)

Arrow(path1,name1)
Arrow(path2,name2)
Arrow(path3,name3)
Arrow(path4,name4)
Arrow(path5,name5)





st_folium(m, width=700, height=500)

