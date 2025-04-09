import numpy as np
import pandas as pd
import re

def preprocess_user_input(user_query):
    filters = {}

    # 🏷 Extracting price (supports "under X lakh", range, lakh/crore)
    price_range_match = re.search(r'(\d+)\s*to\s*(\d+)\s*(lakh|cr|crore)?', user_query, re.IGNORECASE)
    if price_range_match:
        min_price = int(price_range_match.group(1))
        max_price = int(price_range_match.group(2))
        multiplier = 100000  # Default to lakh

        if price_range_match.group(3):  
            if 'cr' in price_range_match.group(3).lower() or 'crore' in price_range_match.group(3).lower():
                multiplier = 10000000  # Convert crore

        filters['Ex-Showroom_Price'] = [min_price * multiplier, max_price * multiplier]

    else:
        under_price_match = re.search(r'under (\d+)\s*(lakh|cr|crore)?', user_query, re.IGNORECASE)
        if under_price_match:
            max_price = int(under_price_match.group(1))
            multiplier = 100000  # Default to lakh

            if under_price_match.group(2):
                if 'cr' in under_price_match.group(2).lower() or 'crore' in under_price_match.group(2).lower():
                    multiplier = 10000000

            filters['Ex-Showroom_Price'] = max_price * multiplier
        else:
            price_match = re.search(r'(\d+)\s*(lakh|cr|crore)?', user_query, re.IGNORECASE)
            if price_match:
                price_value = int(price_match.group(1)) * 100000  # Default to lakh
                if price_match.group(2):
                    if 'cr' in price_match.group(2).lower() or 'crore' in price_match.group(2).lower():
                        price_value *= 100  # Convert crore
                filters['Ex-Showroom_Price'] = price_value

    # 🚗 Extracting seating capacity
    seat_match = re.search(r'(\d+)[- ]?seater', user_query, re.IGNORECASE)
    if seat_match:
        filters['Seating_Capacity'] = int(seat_match.group(1))

    # 🔄 Transmission Handling (Automatic/Manual)
    has_automatic = "automatic" in user_query.lower()
    has_manual = "manual" in user_query.lower()

    if has_automatic and has_manual:
        pass  # If both "automatic" and "manual" are mentioned, don’t filter by transmission
    elif has_automatic:
        filters["Transmission_Automatic"] = 1
    elif has_manual:
        filters["Transmission_Manual"] = 1

    # 🚙 Body Type Filtering
    body_type_mapping = {
        "SUV": "Body_Type_SUV",
        "sedan": "Body_Type_Sedan",
        "hatchback": "Body_Type_Hatchback",
        "MPV": "Body_Type_MPV",
        "MUV": "Body_Type_MUV",
        "coupe": "Body_Type_Coupe",
        "convertible": "Body_Type_Convertible",
        "pickup": "Body_Type_Pick-up",
        "wagon": "Body_Type_Wagon",
        "crossover": "Body_Type_Crossover",
        "sports": "Body_Type_Sports"
    }

    for user_body_type, dataset_column in body_type_mapping.items():
        if re.search(rf'\b{user_body_type}\b', user_query, re.IGNORECASE):
            filters["Body_Type"] = dataset_column
            break

    # Extracting multiple car brands from the query
    makes_in_dataset = ['Tata', 'Datsun', 'Renault', 'Maruti Suzuki', 'Hyundai', 'Premier',
                        'Toyota', 'Nissan', 'Volkswagen', 'Ford', 'Mahindra', 'Fiat',
                        'Honda', 'Force', 'Skoda', 'Jeep', 'Mg', 'Kia', 'Mitsubishi',
                        'Volvo', 'Unknown', 'Mini', 'Bmw', 'Audi', 'Land Rover Rover',
                        'Lexus', 'Jaguar', 'Porsche', 'Land Rover', 'Maserati',
                        'Lamborghini', 'Bentley', 'Ferrari', 'Aston Martin',
                        'Bajaj', 'Icml', 'Isuzu', 'Maruti Suzuki R', 'Dc']  

    found_makes = [brand for brand in makes_in_dataset if brand.lower() in user_query.lower()]

    if found_makes:
        filters["Make"] = found_makes  # ✅ Store as a list, even if one brand is found

    # 🏁 Feature extraction (binary & categorical)
    feature_keywords = {
        'Cruise_Control': ['cruise control'],
        'ABS_(Anti-lock_Braking_System)': ['abs', 'anti-lock braking system'],
        'Airbags': ['airbags', 'safety airbags'],
        'Sunroof': ['sunroof', 'panoramic roof'],
        'Android_Auto': ['android auto'],
        'Apple_CarPlay': ['apple carplay'],
        'Fuel_Type': ['petrol', 'diesel', 'cng', 'electric'],
        'Drivetrain': ['fwd', 'rwd', 'awd', '4wd'],
        'Engine_CC': ['cc', 'engine capacity'],
        'Torque': ['torque'],
        'Horsepower': ['bhp', 'horsepower']
    }

    # 🎯 Handling Mileage Properly (Fixed Range Handling)
    mileage_match = re.search(r'(\d+)\s*to\s*(\d+)\s*kmpl', user_query, re.IGNORECASE)
    if mileage_match:
        mileage_min = int(mileage_match.group(1))
        mileage_max = int(mileage_match.group(2))
        filters["Mileage_(kmpl)"] = [mileage_min, mileage_max]
    else:
        mileage_match = re.search(r'(at least|above|more than|under|below|less than)?\s*(\d+)\s*kmpl', user_query, re.IGNORECASE)
        if mileage_match:
            condition = mileage_match.group(1)
            mileage_value = int(mileage_match.group(2))

            if condition in ["above", "more than", "at least"]:
                filters["Mileage_(kmpl)"] = [mileage_value, float('inf')]
            elif condition in ["under", "below", "less than"]:
                filters["Mileage_(kmpl)"] = [0, mileage_value]
            else:
                filters["Mileage_(kmpl)"] = [mileage_value, float('inf')]

    # 🔧 Extract numerical values for other features
    for feature, keywords in feature_keywords.items():
        for keyword in keywords:
            match = re.search(rf'\b{keyword}\s*(\d+)\b', user_query, re.IGNORECASE)
            if match:
                filters[feature] = int(match.group(1))
            elif re.search(rf'\b{keyword}\b', user_query, re.IGNORECASE):
                if feature in ["Fuel_Type", "Drivetrain"]:
                    filters[feature] = keyword.upper()
                else:
                    filters[feature] = 1

    return filters

def filter_cars(dataset, filters,make_mapping, reverse_make_mapping, reverse_model_mapping, reverse_variant_mapping):
    filtered_df = dataset.copy()
    print("\n🔍 Applying Filters:", filters)
    print("\n🔎 Initial Dataset Size:", dataset.shape)

    # 🚗 Price Filter
    if 'Ex-Showroom_Price' in filters:
        print("\n📊 Filtering by Price Before:", filtered_df.shape)
        if isinstance(filters['Ex-Showroom_Price'], list):
            filtered_df = filtered_df[
                (filtered_df['Ex-Showroom_Price'] >= filters['Ex-Showroom_Price'][0]) &
                (filtered_df['Ex-Showroom_Price'] <= filters['Ex-Showroom_Price'][1])
            ]
        else:
            filtered_df = filtered_df[filtered_df['Ex-Showroom_Price'] <= filters['Ex-Showroom_Price']]
        print("📊 After Price Filter:", filtered_df.shape)

    # 🚗 Make Filter (Re-added)
    if 'Make' in filters:
        print("\n🏷️ Filtering by Make Before:", filtered_df.shape)
        encoded_makes = [make_mapping.get(make, -1) for make in filters['Make']]
        print("🔢 Encoded Makes:", encoded_makes)
        filtered_df = filtered_df[filtered_df['Make'].isin(encoded_makes)]
        print("🏷️ After Make Filter:", filtered_df.shape)

    # 🚙 Body Type Filter
    if 'Body_Type' in filters:
        matching_body_type_columns = [col for col in dataset.columns if filters['Body_Type'] in col]
        if matching_body_type_columns:
            print("📊 Filtering by Body Type Before:", filtered_df.shape)
            filtered_df = filtered_df[(filtered_df[matching_body_type_columns] == 1).any(axis=1)]
            print("📊 After Body Type Filter:", filtered_df.shape)

    # 🚌 Seating Capacity Filter
    if 'Seating_Capacity' in filters:
        print("\n🪑 Filtering by Seating Capacity Before:", filtered_df.shape)
        filtered_df = filtered_df[filtered_df['Seating_Capacity'] == filters['Seating_Capacity']]
        print("🪑 After Seating Capacity Filter:", filtered_df.shape)

    # ⚙️ Transmission Filter
    if 'Transmission_Automatic' in filters and filters['Transmission_Automatic'] == 1:
        print("\n⚙️ Filtering by Transmission (Automatic) Before:", filtered_df.shape)
        filtered_df = filtered_df[filtered_df['Transmission_Automatic'] == 1]
        print("⚙️ After Transmission (Automatic) Filter:", filtered_df.shape)
    
    elif 'Transmission_Manual' in filters and filters['Transmission_Manual'] == 1:
        print("\n⚙️ Filtering by Transmission (Manual) Before:", filtered_df.shape)
        filtered_df = filtered_df[filtered_df['Transmission_Manual'] == 1]
        print("⚙️ After Transmission (Manual) Filter:", filtered_df.shape)

    # ⛽ Fuel Type & Drivetrain Filters
    categorical_features = ['Fuel_Type', 'Drivetrain']
    for feature in categorical_features:
        if feature in filters and feature in dataset.columns:
            print(f"\n⛽ Filtering by {feature} Before:", filtered_df.shape)
            filtered_df = filtered_df[filtered_df[feature].str.upper() == filters[feature].upper()]
            print(f"⛽ After {feature} Filter:", filtered_df.shape)

    # ⛽️ Range-Based Filters (Mileage, Engine CC, Horsepower, Torque)
    range_features = ['ARAI_Certified_Mileage', 'Displacement', 'Power', 'Torque']
    for feature in range_features:
        if feature in filters and feature in dataset.columns:
            print(f"\n📏 Filtering by {feature} Before:", filtered_df.shape)
            if isinstance(filters[feature], list):
                filtered_df = filtered_df[
                    (filtered_df[feature] >= filters[feature][0]) &
                    (filtered_df[feature] <= filters[feature][1])
                ]
            else:
                filtered_df = filtered_df[filtered_df[feature] >= filters[feature]]
            print(f"📏 After {feature} Filter:", filtered_df.shape)

    # ⭐ Binary Feature Filters (Cruise Control, ABS, Sunroof, etc.)
    for feature in filters:
        if feature in dataset.columns and feature not in (
            ['Ex-Showroom_Price', 'Seating_Capacity', 'Fuel_Type', 'Drivetrain',
             'Transmission_Automatic', 'Transmission_Manual', 'Make', 'Body_Type'] + range_features
        ):
            print(f"\n🔘 Filtering by {feature} Before:", filtered_df.shape)
            if isinstance(filters[feature], list):
                filtered_df = filtered_df[filtered_df[feature].isin(filters[feature])]
            else:
                filtered_df = filtered_df[filtered_df[feature] == filters[feature]]
            print(f"🔘 After {feature} Filter:", filtered_df.shape)

    # Convert Encoded Values Back for Display
    filtered_df['Make'] = filtered_df['Make'].map(reverse_make_mapping)
    filtered_df['Model'] = filtered_df['Model'].map(reverse_model_mapping)
    filtered_df['Variant'] = filtered_df['Variant'].map(reverse_variant_mapping)

    # 🚗 Show the Result
    print("\n🚗 Filtered Cars:")
    if not filtered_df.empty:
        print(filtered_df[['Make', 'Model', 'Variant', 'Ex-Showroom_Price']])
    else:
        print("❌ No cars found matching the criteria.")

    return filtered_df  # ✅ Return only the filtered dataset
