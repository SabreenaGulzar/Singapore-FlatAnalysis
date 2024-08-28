import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd

import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)

# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )
    
# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")


# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price]")

    try:
        with st.form("form1"):

            fas_list = [31.0,
                        34.0,
                        35.0,
                        37.0,
                        38.0,
                        39.0,
                        40.0,
                        41.0,
                        42.0,
                        43.0,
                        44.0,
                        45.0,
                        46.0,
                        47.0,
                        48.0,
                        49.0,
                        50.0,
                        51.0,
                        52.0,
                        53.0,
                        54.0,
                        55.0,
                        56.0,
                        57.0,
                        58.0,
                        59.0,
                        60.0,
                        61.0,
                        60.3,
                        63.0,
                        64.0,
                        65.0,
                        66.0,
                        67.0,
                        68.0,
                        69.0,
                        70.0,
                        71.0,
                        72.0,
                        73.0,
                        74.0,
                        75.0,
                        76.0,
                        77.0,
                        78.0,
                        79.0,
                        80.0,
                        81.0,
                        82.0,
                        83.0,
                        84.0,
                        85.0,
                        86.0,
                        87.0,
                        88.0,
                        89.0,
                        90.0,
                        91.0,
                        92.0,
                        93.0,
                        94.0,
                        95.0,
                        96.0,
                        97.0,
                        98.0,
                        99.0,
                        100.0,
                        101.0,
                        102.0,
                        103.0,
                        104.0,
                        105.0,
                        106.0,
                        107.0,
                        108.0,
                        109.0,
                        110.0,
                        111.0,
                        112.0,
                        113.0,
                        114.0,
                        115.0,
                        116.0,
                        117.0,
                        118.0,
                        119.0,
                        120.0,
                        121.0,
                        122.0,
                        123.0,
                        124.0,
                        125.0,
                        126.0,
                        127.0,
                        128.0,
                        129.0,
                        130.0,
                        131.0,
                        132.0,
                        133.0,
                        134.0,
                        135.0,
                        136.0,
                        137.0,
                        138.0,
                        139.0,
                        140.0,
                        141.0,
                        142.0,
                        143.0,
                        144.0,
                        145.0,
                        146.0,
                        147.0,
                        148.0,
                        149.0,
                        150.0,
                        151.0,
                        152.0,
                        153.0,
                        154.0,
                        155.0,
                        156.0,
                        157.0,
                        158.0,
                        159.0,
                        160.0,
                        161.0,
                        162.0,
                        163.0,
                        164.0,
                        165.0,
                        166.0,
                        167.0,
                        168.0,
                        169.0,
                        170.0,
                        171.0,
                        172.0,
                        173.0,
                        174.0,
                        175.0,
                        176.0,
                        177.0,
                        178.0,
                        179.0,
                        180.0,
                        181.0,
                        182.0,
                        183.0,
                        184.0,
                        185.0,
                        186.0,
                        187.0,
                        188.0,
                        189.0,
                        190.0,
                        189.4,
                        192.0,
                        199.0,
                        208.0,
                        210.0,
                        215.0,
                        222.0,
                        237.0,
                        241.0,
                        243.0,
                        249.0,
                        259.0,
                        280.0,
                        62.0,
                        63.1,
                        83.1,
                        100.2]
            floor_area_sqm = st.selectbox('Floor Area (Per Square Meter)', fas_list)

            town_list = ['PUNGGOL', 'SERANGOON', 'SEMBAWANG', 'BISHAN', 'BUKIT TIMAH', 'HOUGANG', 'WOODLANDS', 'PASIR RIS', 'CHOA CHU KANG', 'CENTRAL AREA', 'ANG MO KIO', 'JURONG WEST', 'YISHUN', 'BUKIT BATOK', 'QUEENSTOWN', 'SENGKANG', 'CLEMENTI', 'TAMPINES', 'KALLANG/WHAMPOA', 'GEYLANG', 'JURONG EAST', 'BUKIT MERAH', 'BEDOK', 'MARINE PARADE', 'TOA PAYOH', 'BUKIT PANJANG']
            tl = st.selectbox("Town Name",town_list)

            flat_age_list = [2,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                                20,
                                21,
                                22,
                                23,
                                24,
                                25,
                                26,
                                27,
                                28,
                                29,
                                30,
                                31,
                                32,
                                33,
                                34,
                                35,
                                36,
                                37,
                                38,
                                39,
                                40,
                                41,
                                42,
                                43,
                                44,
                                45,
                                46,
                                47,
                                48,
                                49,
                                50,
                                51,
                                52,
                                53,
                                54,
                                55,
                                56,
                                57,
                                58]
            flat_age = st.selectbox("flat age", flat_age_list)

            lcd_list = [1966,
                        1967,
                        1968,
                        1969,
                        1970,
                        1971,
                        1972,
                        1973,
                        1974,
                        1975,
                        1976,
                        1977,
                        1978,
                        1979,
                        1980,
                        1981,
                        1982,
                        1983,
                        1984,
                        1985,
                        1986,
                        1987,
                        1988,
                        1989,
                        1990,
                        1991,
                        1992,
                        1993,
                        1994,
                        1995,
                        1996,
                        1997,
                        1998,
                        1999,
                        2000,
                        2001,
                        2002,
                        2003,
                        2004,
                        2005,
                        2006,
                        2007,
                        2008,
                        2009,
                        2010,
                        2011,
                        2012,
                        2013,
                        2014,
                        2015,
                        2016,
                        2017,
                        2018,
                        2019,
                        2020,
                        2022]
            lease_commence_date = st.selectbox('Lease Commence Date', lcd_list)

            flat_type_list = ['5 ROOM',
                                '1 ROOM',
                                '2 ROOM',
                                '4 ROOM',
                                'EXECUTIVE',
                                'MULTI-GENERATION',
                                '3 ROOM']
            flat_type = st.selectbox("Flat Type", flat_type_list)

            sr_values = ['01 TO 03',
                            '04 TO 06',
                            '07 TO 09',
                            '10 TO 12',
                            '13 TO 15',
                            '16 TO 18',
                            '19 TO 21',
                            '22 TO 24',
                            '25 TO 27',
                            '28 TO 30',
                            '31 TO 33',
                            '34 TO 36',
                            '37 TO 39',
                            '40 TO 42',
                            '43 TO 45',
                            '46 TO 48',
                            '49 TO 51']
            
            # storey_range = st.selectbox("Storey Range",(sr_values[0],sr_values[1],sr_values[2],sr_values[3],sr_values[4],
            #                               sr_values[5],sr_values[6],sr_values[7],sr_values[8],
            #                               sr_values[9],sr_values[10],sr_values[11],sr_values[12],
            #                               sr_values[13],sr_values[14],sr_values[15],sr_values[16]))
            
            storey_range = st.selectbox("Storey Range",sr_values)


            
            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:

                import pickle

                # Load the model from the file
                with open('RF_model.pkl', 'rb') as file:
                    loaded_rf_model = pickle.load(file)


                new_flat_type = '4-room'
                new_town = 'Ang Mo Kio'

                # Encode the flat_type
                flat_type_mapping = {'flat_type_2 ROOM': [1, 0, 0, 0, 0, 0, 0], 'flat_type_3 ROOM': [0, 1, 0, 0, 0, 0, 0], 'flat_type_4 ROOM': [0, 0, 1, 0, 0, 0, 0], "flat_type_5 ROOM": [0,0,0,1,0,0, 0],
                                    "flat_type_EXECUTIVE": [0,0,0,0,1,0, 0], "flat_type_MULTI-GENERATION": [0,0,0,0,0,1, 0], "flat_type_1 ROOM": [0,0,0,0,0,0,1]}
                flat_type_encoded = flat_type_mapping.get(new_flat_type, [0, 0, 0,0,0,0])  # Default if not found

                # Encode the town
                town_mapping = {
                    'Ang Mo Kio': [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    'Bedok': [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    "PUNGGOL": [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    "SERANGOON": [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    "SEMBAWANG": [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    "BISHAN":[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      "BUKIT TIMAH": [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      "HOUGANG": [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         "WOODLANDS":    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                      "PASIR RIS":       [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        "CHOA CHU KANG":     [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                        "CENTRAL AREA":     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                         "JURONG WEST":    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                          "YISHUN":   [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                           "BUKIT BATOK":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                            "QUEENSTOWN": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                           "SENGKANG":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                           "CLEMENTI":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                            "TAMPINES": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                           "KALLANG/WHAMPOA":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                            "GEYLANG": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                           "JURONG EAST":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,],
                           "BUKIT MERAH":  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,],
                           "MARINE PARADE": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,]
                        #    "TOA PAYOH": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
                    # Add other towns here
                }
                town_encoded = town_mapping.get(new_town, [0] * 30)  # Default if not found

                
                 # -----Calculating median of storey_range to make our calculations quite comfortable-----
                split_list = storey_range.split(' TO ')
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)

                # Combine and predict as before
                input_data = [floor_area_sqm, flat_age, storey_median] + flat_type_encoded + town_encoded
                input_data = np.array(input_data).reshape(1, -1)

                st.write(input_data)

                # Ensure the input data has 33 features
                # assert len(input_data) == 33, "Input data must have 33 features"

# Convert to a NumPy array 
                input_data = np.array(input_data).reshape(1, -1)

                prediction = loaded_rf_model.predict(input_data)[0]

                print(f"Predicted resale price: {prediction}")
                st.write('## :green[Predicted resale price:] ', prediction)
                

                


    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat", e)