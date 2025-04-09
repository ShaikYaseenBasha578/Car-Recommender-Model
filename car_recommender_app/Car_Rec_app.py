import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from utils import preprocess_user_input, filter_cars

# --- Apply Dark Theme Styling ---
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: white;
    }
    .stTextInput, .stTextInput input {
        background-color: #333333 !important;
        color: white !important;
        border-radius: 8px;
        padding: 10px;
    }
    .stTextInput input:focus {
        box-shadow: 0px 0px 5px rgba(255, 255, 255, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model and Data ---
st.markdown("<h1 style='text-align: center; color: white;'>Car Recommender 🚗</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    with open('car_recommendation_assets/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    with open('car_recommendation_assets/categorical_mappings.pkl', 'rb') as f:
        categorical_mappings = pickle.load(f)

    return label_encoders, categorical_mappings

@st.cache_data
def load_data():
    return pd.read_csv("car_recommendation_assets/processed_dataset.csv")

label_encoders, categorical_mappings = load_assets()
df = load_data()

# --- User Input ---
st.subheader("Describe what you're looking for")
user_query = st.text_input(
    "Describe what you're looking for",
    placeholder="e.g., 'Automatic SUV under 15 lakhs with cruise control'",
    help="Type your query to find the best car recommendations!"
)

# --- Run Recommendation ---
if user_query:
    try:
        # ✅ Extract filters
        filters = preprocess_user_input(user_query)

        # ✅ Ensure necessary mappings
        required_keys = [
            'make_mapping', 'model_mapping', 'variant_mapping',
            'reverse_make_mapping', 'reverse_model_mapping', 'reverse_variant_mapping'
        ]
        for key in required_keys:
            if key not in categorical_mappings:
                raise KeyError(f"Missing '{key}' in categorical_mappings.")

        make_mapping = categorical_mappings['make_mapping']
        model_mapping = categorical_mappings['model_mapping']
        variant_mapping = categorical_mappings['variant_mapping']
        reverse_make_mapping = categorical_mappings['reverse_make_mapping']
        reverse_model_mapping = categorical_mappings['reverse_model_mapping']
        reverse_variant_mapping = categorical_mappings['reverse_variant_mapping']

        # ✅ Filter dataset with proper arguments
        filtered_df = filter_cars(
            df, 
            filters,
            make_mapping=make_mapping,
            reverse_make_mapping=reverse_make_mapping,
            reverse_model_mapping=reverse_model_mapping,
            reverse_variant_mapping=reverse_variant_mapping
        )

        if filtered_df.empty:
            st.warning("No cars found matching your query.")
        else:
            st.subheader("Filtered Dataset")
            st.dataframe(filtered_df)

            # ✅ Normalize numerical features
            numerical_cols = ['Ex-Showroom_Price', 'Displacement', 'Power', 'ARAI_Certified_Mileage']
            scaler = StandardScaler()
            filtered_df[numerical_cols] = scaler.fit_transform(filtered_df[numerical_cols])

            # ✅ Extract categorical one-hot encoding
            encoded_cats = filtered_df.filter(like='_').values
            final_features = np.hstack((filtered_df[numerical_cols], encoded_cats))

            # ✅ Train KNN on filtered dataset
            knn_model = NearestNeighbors(n_neighbors=min(10, len(filtered_df)), metric='euclidean')
            knn_model.fit(final_features)

            # ✅ Convert query to feature vector
            query_vector = np.zeros(len(numerical_cols))
            query_categorical = np.zeros(encoded_cats.shape[1])
            one_hot_columns = list(df.filter(like='_').columns)

            for col in filters:
                if col in numerical_cols:
                    query_vector[numerical_cols.index(col)] = np.mean(filters[col]) if isinstance(filters[col], list) else filters[col]
                elif col in df.columns:
                    cat_column_name = col + "_" + str(filters[col])
                    if cat_column_name in one_hot_columns:
                        cat_index = one_hot_columns.index(cat_column_name)
                        query_categorical[cat_index] = 1

            full_query_vector = np.hstack((query_vector, query_categorical))

            # ✅ Get recommendations
            distances, indices = knn_model.kneighbors(full_query_vector.reshape(1, -1))

            # ✅ Display Results
            st.markdown("<h2 style='color: white;'>Top Car Recommendations:</h2>", unsafe_allow_html=True)
            selected_cars = []
            make_count = {}
            total_printed = 0
            max_recommendations = 5
            max_per_make = 2
            
            for index, distance in zip(indices[0], distances[0]):
                if total_printed >= max_recommendations:
                    break  
                
                car = filtered_df.iloc[index]
                make_name = car['Make']
                
                if make_count.get(make_name, 0) >= max_per_make:
                    continue  
                
                make_count[make_name] = make_count.get(make_name, 0) + 1
                selected_cars.append(car)
                total_printed += 1  

            for car in selected_cars:
                price = car['Ex-Showroom_Price'] * scaler.scale_[0] + scaler.mean_[0]
                mileage = car['ARAI_Certified_Mileage'] * scaler.scale_[3] + scaler.mean_[3]
                displacement = car['Displacement'] * scaler.scale_[1] + scaler.mean_[1]
                power = car['Power'] * scaler.scale_[2] + scaler.mean_[2]
                # ✅ Extract Transmission from one-hot columns
                transmission_type = 'Not specified'
                for col in car.index:
                    if col.startswith('Transmission_') and car[col] == 1:
                        transmission_type = col.replace('Transmission_', '')
                        transmission_type = transmission_type.replace('_', ' ').title()
                        break

                st.markdown(f"""
                <span style='font-size:18px; font-weight:bold; color:white;'>
                🚗 {car['Make']} {car['Model']} {car['Variant']} – ₹{price:,.0f}
                </span>
                """, unsafe_allow_html=True)

                with st.expander("View Details"):
                    st.markdown(f"""
                    <div style="background-color:#1e1e1e; padding:10px; border-radius:10px;">
                        <p>💰 <strong>Price:</strong> ₹{price:,.0f}</p>
                        <p>⛽ <strong>Mileage:</strong> {mileage:.1f} kmpl</p>
                        <p>⚙️ <strong>Engine:</strong> {displacement:.1f} cc</p>
                        <p>🌀 <strong>Power:</strong> {power:.0f} BHP</p>
                        <p>🧰 <strong>Transmission:</strong> {transmission_type}</p>
                        <p>👥 <strong>Seating:</strong> {int(car.get('Seating_Capacity', 0))} Seater</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # ✅ Optional direct comparison with similar cars
                    st.markdown("**🔍 Similar cars from same Make:**")
                    similar_cars_encoded = filtered_df[
                        (filtered_df['Make'] == car['Make']) &
                        (filtered_df['Model'] != car['Model']) &
                        (filtered_df['Variant'] != car['Variant'])
                    ].head(3)

                    if similar_cars_encoded.empty:
                        st.write("No similar cars found.")
                    else:
                        for _, sim_car in similar_cars_encoded.iterrows():
                            sim_make = sim_car['Make']
                            sim_model = sim_car['Model']
                            sim_variant = sim_car['Variant']
                            sim_price = sim_car['Ex-Showroom_Price'] * scaler.scale_[0] + scaler.mean_[0]

                            st.markdown(f"- {sim_make} {sim_model} {sim_variant} – ₹{sim_price:,.0f}")

    except Exception as e:
        st.error(f"❌ Error: {e}")
