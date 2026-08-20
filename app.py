from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="California Property Close Price Predictor",
    page_icon="🏠",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

PREPROCESSOR_PATH = MODELS_DIR / "week7_preprocessor.joblib"
MODEL_PATH = MODELS_DIR / "best_week7_xgboost.joblib"


# ---------------------------------------------------------
# Load saved Week 7 artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        missing = []
        if not PREPROCESSOR_PATH.exists():
            missing.append(str(PREPROCESSOR_PATH))
        if not MODEL_PATH.exists():
            missing.append(str(MODEL_PATH))

        raise FileNotFoundError(
            "Missing required Week 7 model file(s):\n"
            + "\n".join(missing)
            + "\n\nRun Week 7 first and keep the saved files inside the models/ folder."
        )

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    return preprocessor, model


# ---------------------------------------------------------
# Feature engineering used in Weeks 6–7
# ---------------------------------------------------------
def safe_ratio(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    if numerator is None or pd.isna(numerator):
        return np.nan
    return numerator / denominator


def build_property_row(
    living_area,
    bedrooms,
    bathrooms,
    lot_size,
    year_built=np.nan,
    original_list_price=np.nan,
    latitude=np.nan,
    longitude=np.nan,
    garage_spaces=np.nan,
    stories=np.nan,
    county=np.nan,
    city=np.nan,
    postal_code=np.nan,
    architectural_style=np.nan,
    property_condition=np.nan,
    district_name=np.nan,
):
    current_year = pd.Timestamp.today().year

    property_age = (
        current_year - year_built
        if pd.notna(year_built)
        else np.nan
    )

    return pd.DataFrame(
        [
            {
                "LivingArea": living_area,
                "BedroomsTotal": bedrooms,
                "BathroomsTotalInteger": bathrooms,
                "LotSizeSquareFeet": lot_size,
                "YearBuilt": year_built,
                "OriginalListPrice": original_list_price,
                "Latitude": latitude,
                "Longitude": longitude,
                "GarageSpaces": garage_spaces,
                "Stories": stories,
                "PropertyAge": property_age,
                "BedBathRatio": safe_ratio(bedrooms, bathrooms),
                "LivingAreaPerBedroom": safe_ratio(living_area, bedrooms),
                "LotToLivingRatio": safe_ratio(lot_size, living_area),
                "CountyOrParish": county,
                "City": city,
                "PostalCode": postal_code,
                "ArchitecturalStyle": architectural_style,
                "PropertyCondition": property_condition,
                "DistrictName": district_name,
            }
        ]
    )


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🏠 California Property Close Price Predictor")

st.write(
    "Enter basic property information to estimate the final sale price "
    "using the tuned Week 7 XGBoost model."
)

st.caption(
    "This is an internship demonstration model, not a professional appraisal."
)

try:
    preprocessor, model = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()


st.subheader("Property Information")

col1, col2 = st.columns(2)

with col1:
    living_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=100_000,
        value=2_000,
        step=100,
    )

    bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=30,
        value=3,
        step=1,
    )

with col2:
    bathrooms = st.number_input(
        "Bathrooms",
        min_value=0,
        max_value=30,
        value=2,
        step=1,
    )

    lot_size = st.number_input(
        "Lot Size (sq ft)",
        min_value=0,
        max_value=10_000_000,
        value=6_000,
        step=500,
    )


# The original Week 7 model was trained with more than four raw variables.
# These fields are optional because the saved preprocessor can impute missing
# values. Providing them can make the input closer to the training schema.
with st.expander("Optional property details"):
    original_list_price_input = st.number_input(
        "Original List Price ($) — optional",
        min_value=0,
        max_value=500_000_000,
        value=0,
        step=10_000,
    )

    year_built_input = st.number_input(
        "Year Built — optional",
        min_value=0,
        max_value=pd.Timestamp.today().year,
        value=0,
        step=1,
    )

    garage_spaces_input = st.number_input(
        "Garage Spaces — optional",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    stories_input = st.number_input(
        "Stories — optional",
        min_value=0,
        max_value=20,
        value=0,
        step=1,
    )

    city_input = st.text_input("City — optional").strip()
    county_input = st.text_input("County — optional").strip()
    postal_code_input = st.text_input("Postal Code — optional").strip()
    district_input = st.text_input("School District — optional").strip()
    style_input = st.text_input("Architectural Style — optional").strip()
    condition_input = st.text_input("Property Condition — optional").strip()

    latitude_input = st.number_input(
        "Latitude — optional (enter 0 if unknown)",
        min_value=-90.0,
        max_value=90.0,
        value=0.0,
        format="%.6f",
    )

    longitude_input = st.number_input(
        "Longitude — optional (enter 0 if unknown)",
        min_value=-180.0,
        max_value=180.0,
        value=0.0,
        format="%.6f",
    )


if st.button("Predict Close Price", type="primary", use_container_width=True):
    original_list_price = (
        float(original_list_price_input)
        if original_list_price_input > 0
        else np.nan
    )
    year_built = (
        float(year_built_input)
        if year_built_input > 0
        else np.nan
    )
    garage_spaces = (
        float(garage_spaces_input)
        if garage_spaces_input > 0
        else np.nan
    )
    stories = (
        float(stories_input)
        if stories_input > 0
        else np.nan
    )

    latitude = (
        float(latitude_input)
        if latitude_input != 0
        else np.nan
    )
    longitude = (
        float(longitude_input)
        if longitude_input != 0
        else np.nan
    )

    property_df = build_property_row(
        living_area=float(living_area),
        bedrooms=float(bedrooms),
        bathrooms=float(bathrooms),
        lot_size=float(lot_size),
        year_built=year_built,
        original_list_price=original_list_price,
        latitude=latitude,
        longitude=longitude,
        garage_spaces=garage_spaces,
        stories=stories,
        county=county_input if county_input else np.nan,
        city=city_input if city_input else np.nan,
        postal_code=postal_code_input if postal_code_input else np.nan,
        architectural_style=style_input if style_input else np.nan,
        property_condition=condition_input if condition_input else np.nan,
        district_name=district_input if district_input else np.nan,
    )

    try:
        transformed = preprocessor.transform(property_df)
        prediction = float(model.predict(transformed)[0])
        prediction = max(prediction, 0.0)

        st.success("Prediction complete")
        st.metric(
            "Estimated Close Price",
            f"${prediction:,.0f}",
        )

        # Week 8 showed that a few extreme predictions can dominate squared error.
        # Warn instead of silently changing the model output.
        unusual = prediction > 50_000_000

        if pd.notna(original_list_price):
            if prediction > 5 * original_list_price:
                unusual = True

        if unusual:
            st.warning(
                "This prediction is unusually large. Week 8 found that the "
                "XGBoost model can occasionally produce extreme outlier "
                "predictions, so this result should be reviewed carefully."
            )

        with st.expander("View model input"):
            display_df = property_df.copy()
            display_df = display_df.replace({np.nan: "Missing / imputed"})
            st.dataframe(display_df.T, use_container_width=True)

    except Exception as exc:
        st.error(
            "The prediction could not be generated. "
            "Please confirm that the Week 7 model and preprocessor match this app."
        )
        st.exception(exc)


st.divider()

st.caption(
    "Week 9 — Optional Streamlit Prediction App | "
    "California Property Close Price Prediction"
)
