import folium
import streamlit as st
import ee
import random
import numpy as np
import os
import leafmap.foliumap as leafmap
import requests
import datetime
import pickle
import pandas as pd
import plotly.express as px
import time
import json
from folium.plugins import Draw
import streamlit_folium
from folium.plugins import Draw
from datetime import datetime
from datetime import date
import json

#st.set_page_config(layout="wide")

# json_data=st.secrets["json_data"]
# service_account = st.secrets["service_account"]


# json_object = json.loads(json_data, strict=False)
# json_object = json.dumps(json_object)
# credentials = ee.ServiceAccountCredentials(service_account, key_data=json_object)
# ee.Initialize(credentials)

ee.Authenticate()
ee.Initialize(project='vegetation-2023-408901')

def load_model(path):
    with open (path, 'rb') as loaded_model:
        model = pickle.load(loaded_model)
    return model

def get_steps(steps):
  benchmark_year=2023
  today=str(date.today())
  year=today.split('-')[0]

  month=today.split('-')[1]
  gap=int(year)-benchmark_year
  gap_month=gap*12 + int(month)
  total_steps=gap_month+steps
  return total_steps

def generate_timestamps(num_steps):
    months = 12
    year = 2023
    month_ = 1

    dates = []

    for num in range(num_steps):
        year_change = num // 12
        year_ = year + year_change

        if num %12 == 0:
            month_ = 1

        date = f"{year_}-{month_}-01"
        dates.append(date)

        month_ += 1
    return dates

def predict(model, num_steps,):
    prediction=model.forecast(num_steps)
    timestamp=generate_timestamps(num_steps)
    data_dict={
        'predictions':list(prediction.values),
        #'real_values': np.random.random(num_steps),
        'timesteps': timestamp,
    }
    
    df=pd.DataFrame(data_dict)
    fig=px.line(df, x="timesteps", y=df.columns[0:2], title='Predicted Mean NDVI Trend', width=1000, height=700)
    return fig

def app():
    st.title("NDVI Displayer")


    SA_CENTER=[39.204260, -120.755200]
    SJ_CENTER=[36.824278, -118.910522]

    valleys = ['Sacramento', 'San Joaquin']
    model=None
    sa_model = load_model('sa_model')
    sj_model = load_model("sj_model")

    st.write("Select which valley your field is in:")

    st.session_state['valleySelect'] = st.selectbox('Select a valley', valleys)

    start_date_input = st.date_input('Start date (YYY-MM-DD)', value="default_value_today", format="YYYY-MM-DD")
    print(type(start_date_input))
    end_date_input = st.date_input('End date (YYY-MM-DD)', value="default_value_today", format="YYYY-MM-DD")
    #final_steps=get_steps(slider_val)

    if st.session_state['valleySelect']=='Sacramento':
        model=sa_model
        loc=SA_CENTER # recenters the map to SA valley
        st.session_state['valleySelect']='Sacramento'

    else:
        model=sj_model
        loc=SJ_CENTER # recenters the map to SJ Valley
        st.session_state['valleySelect']='San Joaquin'
    print('loc:', loc)


    m = folium.Map(location=loc,tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr = 'Esri', zoom_start=9)

    st.session_state['m'] = m

    Draw(export=False).add_to(m)

    output = streamlit_folium(m, width=1500, height=500)
    st.session_state['output'] = output
    print('output is:',output)
    if output is not None:
        if output['all_drawings'] is not None:
            if output['all_drawings'][0] is not None:
                if output['all_drawings'][0]['geometry'] is not None:
                    st.session_state['coords'] = output['all_drawings'][0]['geometry']['coordinates'] # COORDINATES FOR SELECTED ROI
                    coordinates = []
                    coordinates = st.session_state['coords']
                    geojson_object = {
                        'type': 'Polygon',
                        'coordinates': coordinates
                    }
                    #st.write('type(geojson_object):', type(geojson_object))
                    #st.write('geojson_object:', geojson_object)

                    roi=ee.Geometry(geojson_object)
                    #st.write('roi:', roi)
                    #st.write('roi.getInfo():', roi.getInfo())

                    # # GeoJSON strings need to be converted to an object.
                    # geojson_string = json.dumps(geojson_object)
                    # print('A GeoJSON string needs to be converted to an object:',
                    #       ee.Geometry(json.loads(geojson_string)).getInfo())

                    # # Use ee.Geometry to cast computed geometry objects into the ee.Geometry
                    # # class to access their methods. In the following example an ee.Geometry
                    # # object is stored as a ee.Feature property. When it is retrieved with the
                    # # .get() function, a computed geometry object is returned. Cast the computed
                    # # object as a ee.Geometry to get the geometry's bounds, for instance.
                    # feature = ee.Feature(None, {'geom': ee.Geometry(geojson_object)})
                    # print('Cast computed geometry objects to ee.Geometry class:',
                    #       ee.Geometry(feature.get('geom')).bounds().getInfo())

                    # #st.write(output)


                    def collect(region, start_date, end_date):
                        #print("Region:", region.getInfo())
                        roi = region


                        def printType(prompt, object):
                            print(prompt, type(object))
                        def format_date(timestamp):
                            """
                            Convert the UTC timestamps to date time

                            @parameters
                            timestamp: UTC timestamps in milliseconds

                            @return
                            None
                            """
                            # get the seconds by dividing 1000
                            #print(timestamp)
                            timestamp = timestamp/1000
                            # Convert the UTC timestamp to a datetime object
                            datetime_object = datetime.utcfromtimestamp(timestamp)
                            # Format the datetime object as a string (optional)
                            formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S UTC")
                            #print("Formatted Datetime:", formatted_datetime)
                            return formatted_datetime
                        def print_dict(dictionary):
                            for k, v in dictionary.items():
                                print(k, v)
                        def mapFunctionNDVI(specific_image):
                            Red = specific_image.select('B4')
                            NIR = specific_image.select('B8')
                            NDVI_temp = specific_image
                            NDVI = NDVI_temp.addBands(((NIR.subtract(Red)).divide(NIR.add(Red))).rename('NDVI'))  #ee.Image
                            #nameOfBands = NDVI.bandNames().getInfo()
                            #nameOfBands.remove("B2")
                            #print(nameOfBands) # Check if everything in order

                            NDVI = NDVI.select('NDVI') # Select all bands except the one you wanna remove
                            #NDVI.copyProperties(specific_image)
                            return NDVI
                        def calculateNDVIStatsForImage(ndvi_image):
                            #image = sentinel2ImageCollection.first()
                            #print('Image type is :', type(ndvi_image))

                            reducers = ee.Reducer.min() \
                            .combine(
                            ee.Reducer.max(),
                            sharedInputs = True
                            ).combine(
                            ee.Reducer.mean(),
                            sharedInputs = True
                            ).combine(
                            ee.Reducer.stdDev(),
                            sharedInputs = True
                            )

                            multi_stats = ndvi_image.reduceRegion(
                                reducer=reducers,
                                geometry=roi,
                                scale=30,
                                crs='EPSG:32610'
                            )

                            return ndvi_image.set('stats', multi_stats.values())
                        def calculateNDVIStatsForImageAsDictionary(ndvi_image):
                            #image = sentinel2ImageCollection.first()
                            #print('Image type is :', type(ndvi_image))

                            reducers = ee.Reducer.min() \
                            .combine(
                            ee.Reducer.max(),
                            sharedInputs = True
                            ).combine(
                            ee.Reducer.mean(),
                            sharedInputs = True
                            ).combine(
                            ee.Reducer.stdDev(),
                            sharedInputs = True
                            )

                            multi_stats = ndvi_image.reduceRegion(
                                reducer=reducers,
                                geometry=roi,
                                scale=30,
                                crs='EPSG:32610'
                            )
                            #dateStart = format_date(ndvi_image.get('system:time_start').getInfo())

                            #multi_stats.set('dateStart','something')
                            return ndvi_image.set('stats_dictionary', multi_stats)


                        WANTED_BANDS = ['B2', 'B3', 'B4', 'B8']
                        
                        print('ROI:', roi)
                        print('start_date:', start_date)
                        print('end_date:', end_date)
                        sentinel2ImageCollection = (
                            ee.ImageCollection('COPERNICUS/S2')
                            .select(WANTED_BANDS)
                            .filterBounds(roi)
                            .filterDate(start_date, end_date)
                            .filter(
                                ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE',10)
                                )
                            #.limit(2)
                        )
                        ndviCollection = sentinel2ImageCollection.map(mapFunctionNDVI) #collection of ndvi sentinel images
                        statsCollection = ndviCollection.map(calculateNDVIStatsForImage)
                        statsList = statsCollection.toList(statsCollection.size())
                        statsLength = statsList.length().getInfo()
                        image = statsCollection.first()
                        dateStart = format_date(image.get('system:time_start').getInfo())
                        #print('DONE!')


                        dictionaryCollection = ndviCollection.map(calculateNDVIStatsForImageAsDictionary)
                        dictionaryList = dictionaryCollection.toList(dictionaryCollection.size())
                        dictionaryLength = dictionaryList.length().getInfo()
                        featureList = []
                        #print('Start time =', datetime.now())
                        for index in range(dictionaryLength):
                            image = ee.Image(dictionaryList.get(index))
                            dateStart = format_date(image.get('system:time_start').getInfo())
                            #print('dateStart:',dateStart)
                            dictionary = image.get('stats_dictionary').getInfo()
                            dictionary['dateStart'] = dateStart

                            feature = ee.Feature(None, dictionary)
                            featureList.append(feature)

                        featureCollection = ee.FeatureCollection(featureList)
                        #print('featureList:', featureList[0])
                        #st.write(featureCollection)
                        #print()
                        #print('Task Started')
                        #print('End =', datetime.now())
                        return featureList
                    #st.write('roi before feature call', roi) 
                    start_date = start_date_input.strftime("%Y-%m-%d")
                    #st.write('start_date:', start_date)
                    #st.write('type(start_date):', type(start_date))
                    end_date = end_date_input.strftime("%Y-%m-%d") 
                    if start_date is not None and end_date is not None:
                        feature_list = collect(roi, start_date, end_date)
                        #feature_list = collect(roi, '2023-05-01', '2023-06-01')
                        #st.write('feature_list', feature_list)  
                        date_list=[]
                        mean_list=[]

                        for feature in feature_list:
                            #st.write(feature.getInfo())
                            #st.write(type(feature))
                            #st.write('')
                            date_list.append(feature.get("dateStart").getInfo())
                            mean_list.append(feature.get("NDVI_mean").getInfo())
                        #st.write('date_list:', date_list)
                        #st.write('mean_list:', mean_list)
                        #st.write(feature.getInfo())
                        mean_df = pd.DataFrame(list(zip(date_list, mean_list)))
                        mean_df.columns=['date','NDVI_mean']
                        print('mean_df:', mean_df)
                        fig=px.line(mean_df, x='date', y='NDVI_mean', title='Test Plot', width=700, height=500)
                        st.plotly_chart(fig)
                        slider_val = st.slider("Select a step value", min_value=1, max_value=24,step=1,help='Select a step value')
                        final_steps=get_steps(slider_val)
                        if st.session_state['valleySelect']=='Sacramento':
                            model=sa_model
                            fig2=predict(sa_model,final_steps)
                        else:
                            model=sj_model
                            fig2=predict(sj_model,final_steps)
                        st.plotly_chart(fig2)