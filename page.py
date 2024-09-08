import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import joblib
import statistics
import numpy as np
from geopy.distance import geodesic
import json
import requests
import warnings
import streamlit_lottie as st_lottie

# Suppress warnings
warnings.filterwarnings("ignore")


# Function to load Lottie animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


# Load Lottie animation from a file
lottie_animation = load_lottie("C:/Users/visha/Downloads/Animation - 1725605584607.json")

# Load MRT station location data
mrt_location = pd.read_csv("C:/Users/visha/Downloads/mrt_location.xls")

# Image path for the sidebar
image_path = "C:/Users/visha/Downloads/pngwing.com (1).png"

# Display Lottie animation in the main area of the app
st_lottie.st_lottie(lottie_animation, height=200, key="animation")

# Display a markdown title for the application
st.markdown('<h1 style="color: red;">Singapore resale price prediction</h1>', unsafe_allow_html=True)

# Add a horizontal line for separation
with st.container():
    st.write('---')

# Sidebar configuration
with st.sidebar:
    # Display the image in the sidebar
    st.image(image_path, use_column_width=True)

    # Sidebar menu for selecting options
    selected = option_menu("Prediction", ["Selling Price"],
                           icons=['currency-dollar'], menu_icon="robot", default_index=0)

    # Sidebar caption with instructions and description of the app
    st.caption("""
        ## Welcome to the Singapore Resale Price Prediction App!

        This application predicts the resale price of properties in Singapore based on various input parameters. Whether you are a property buyer, seller, or investor, this tool helps you get an estimate of the resale value of a flat.

        ### How to Use:
        1. **Street Name**: Enter the street name of the property.
        2. **Block Number**: Provide the block number of the property.
        3. **Floor Area (Per Square Meter)**: Enter the floor area in square meters.
        4. **Lease Commence Date**: Input the year when the lease commenced.
        5. **Storey Range**: Provide the storey range of the property (e.g., '10 TO 15').

        After entering these details, click the **Submit** button to get an estimated resale price. The app also displays the location on the map.

        ### Contact Us:
        If you have any questions or need support, feel free to reach out to us at [support@example.com](mailto:support@example.com).
    """)

# Load the pre-trained transformer and Random Forest model
transformer = joblib.load('power_transform.pkl')
random_forest_model = joblib.load('singapur_random_forest_model.pkl')

# Check if the "Selling Price" option is selected
if selected == "Selling Price":
    # Collect user inputs for prediction
    street_name = st.text_input("Street Name")
    block = st.text_input("Block Number")
    floor_area_sqm = st.text_input('Floor Area (Per Square Meter)')
    lease_commence_date = st.number_input('Lease Commence Date')
    storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

    # Submit button to make predictions
    if st.button('Submit'):
        # Concatenate block and street name for geolocation
        address = block + " " + street_name

        # -----Calculating remaining lease years based on lease commencement date-----
        lease_remain_years = 99 - (2023 - lease_commence_date)

        # -----Calculate the median value of the storey range provided by the user-----
        split_list = storey_range.split(' TO ')  # Split the input range
        float_list = [float(i) for i in split_list]  # Convert to float for calculation
        storey_median = statistics.median(float_list)  # Find the median of the range

        # -----Use OneMap API to get latitude and longitude of the property-----
        query_string = 'https://www.onemap.gov.sg/api/common/elastic/search?searchVal=' + str(
            address) + '&returnGeom=Y&getAddrDetails=Y&pageNum=1'
        resp = requests.get(query_string)
        origin = []
        data_geo_location = json.loads(resp.content)

        # Check if the location data is found
        if data_geo_location['found'] != 0:
            # Extract latitude and longitude from the response
            latitude = data_geo_location['results'][0]['LATITUDE']
            longitude = data_geo_location['results'][0]['LONGITUDE']
            origin.append((latitude, longitude))

            # Create a DataFrame for displaying on the map
            data = {
                'latitude': [float(latitude)],
                'longitude': [float(longitude)]
            }
            df = pd.DataFrame(data)

            # -----Appending MRT station coordinates to a list-----
            mrt_lat = mrt_location['latitude']
            mrt_long = mrt_location['longitude']
            list_of_mrt_coordinates = [(lat, long) for lat, long in zip(mrt_lat, mrt_long)]

            # -----Calculating distance to nearest MRT station-----
            list_of_dist_mrt = [geodesic(origin, destination).meters for destination in list_of_mrt_coordinates]
            min_dist_mrt = min(list_of_dist_mrt)  # Get the minimum distance to the nearest MRT station
            list_of_dist_mrt.clear()

            # -----Calculate distance from Central Business District (CBD)-----
            cbd_dist = geodesic(origin, (1.2830, 103.8513)).meters  # CBD coordinates

            # -----Prepare the input features for the model-----
            new_sample = np.array([[latitude, longitude, cbd_dist, min_dist_mrt, np.log(int(floor_area_sqm)),
                                    np.log(lease_commence_date), lease_remain_years, np.log(storey_median)]])

            # -----Make prediction using the pre-trained Random Forest model-----
            y_pred = random_forest_model.predict(new_sample)

            # -----Transform the predicted value back to the original scale-----
            y_pred_original = int(transformer.inverse_transform(y_pred.reshape(-1, 1)))

            # Display the prediction and map
            st.write('---')
            st.markdown(f"""
                    🏢 **Street Name**: {street_name}  
                    🏢 **Block Number**: {block}  
                    📐 **Floor Area**: {floor_area_sqm} sqm  
                    📅 **Lease Commence Date**: {lease_commence_date}  
                    🏙️ **Storey Range**: {storey_range}
                    """)
            st.write(f"""💵 **Predicted Resale Price**: ${y_pred_original:,}""")

            # Display the map with the property's location
            st.map(df, zoom=16, size=8, use_container_width=True)
