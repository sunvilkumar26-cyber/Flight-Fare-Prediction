
import pickle
import re
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --------------------------------------------------
# 1. LOAD TRAINED MODEL PACKAGE
# --------------------------------------------------

with open("flight_fare_model.pkl", "rb") as file:
    package = pickle.load(file)

preprocessor = package["preprocessor"]
model = package["model"]

categorical_features = package["categorical_features"]
numerical_features = package["numerical_features"]

# Get dropdown categories from the trained encoder
encoder = preprocessor.named_transformers_["cat"]

dropdowns = dict(
    zip(categorical_features, encoder.categories_)
)

# --------------------------------------------------
# 2. AIRPORT CODE MAPPING
# --------------------------------------------------

airport_codes = {
    "Banglore": "BLR",
    "Bangalore": "BLR",
    "Chennai": "MAA",
    "Delhi": "DEL",
    "New Delhi": "DEL",
    "Kolkata": "CCU",
    "Mumbai": "BOM",
    "Cochin": "COK",
    "Hyderabad": "HYD"
}

stops_mapping = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4
}

# --------------------------------------------------
# 3. DURATION CONVERSION
# --------------------------------------------------

def convert_duration(duration):

    hours = 0
    minutes = 0

    hour_match = re.search(r"(\d+)\s*h", duration)
    minute_match = re.search(r"(\d+)\s*m", duration)

    if hour_match:
        hours = int(hour_match.group(1))

    if minute_match:
        minutes = int(minute_match.group(1))

    if not hour_match and not minute_match:
        raise ValueError("Enter duration like 2h 50m or 45m.")

    if minutes >= 60:
        raise ValueError("Duration minutes must be below 60.")

    return hours * 60 + minutes


# --------------------------------------------------
# 4. FLIGHT INPUT VALIDATION
# --------------------------------------------------

def validate_flight(data):

    source = data["source"]
    destination = data["destination"]
    route = data["route"]
    total_stops = data["total_stops"]

    if source not in airport_codes:
        raise ValueError("Invalid source airport.")

    if destination not in airport_codes:
        raise ValueError("Invalid destination airport.")

    if source == destination:
        raise ValueError("Source and destination cannot be the same.")

    if total_stops not in stops_mapping:
        raise ValueError("Invalid number of stops.")

    # Split route into individual airport codes
    route_parts = [
        part.strip()
        for part in route.split("→")
    ]

    if len(route_parts) < 2:
        raise ValueError("Please select a valid route.")

    expected_source = airport_codes[source]
    expected_destination = airport_codes[destination]

    # Check route starting airport
    if route_parts[0] != expected_source:
        raise ValueError(
            f"Route must start at {expected_source} for {source}."
        )

    # Check route ending airport
    if route_parts[-1] != expected_destination:
        raise ValueError(
            f"Route must end at {expected_destination} for {destination}."
        )

    # Validate number of stops
    route_stops = len(route_parts) - 2

    if route_stops != stops_mapping[total_stops]:
        raise ValueError(
            "Selected route does not match the number of stops."
        )

    # Validate duration against clock times
    departure = datetime.strptime(
        data["departure_time"], "%H:%M"
    )

    arrival = datetime.strptime(
        data["arrival_time"], "%H:%M"
    )

    duration_minutes = convert_duration(data["duration"])

    elapsed_minutes = (
        (arrival.hour * 60 + arrival.minute)
        - (departure.hour * 60 + departure.minute)
    ) % 1440

    if duration_minutes % 1440 != elapsed_minutes:
        raise ValueError(
            "Duration does not match departure and arrival times. "
            "For flights longer than 24 hours, the time fields alone "
            "cannot confirm the full elapsed duration."
        )

    return duration_minutes


# --------------------------------------------------
# 5. FLASK HOME PAGE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None
    form_data = {}

    if request.method == "POST":

        form_data = request.form.to_dict()

        try:

            # Validate journey date
            journey_date = datetime.strptime(
                form_data["journey_date"], "%Y-%m-%d"
            )

            # Validate flight details
            duration_minutes = validate_flight(form_data)

            departure = datetime.strptime(
                form_data["departure_time"], "%H:%M"
            )

            arrival = datetime.strptime(
                form_data["arrival_time"], "%H:%M"
            )

            # Prepare model input
            input_data = {

                "Airline": form_data["airline"],
                "Source": form_data["source"],
                "Destination": form_data["destination"],
                "Route": form_data["route"],
                "Additional_Info": form_data["additional_info"],

                "Journey_Day": journey_date.day,
                "Journey_Month": journey_date.month,

                "Departure_Hour": departure.hour,
                "Departure_Minute": departure.minute,

                "Arrival_Hour": arrival.hour,
                "Arrival_Minute": arrival.minute,

                "Duration_Minutes": duration_minutes,

                "Total_Stops_Numeric": stops_mapping[
                    form_data["total_stops"]
                ]
            }

            input_df = pd.DataFrame([input_data])

            # Keep the same feature order used during training
            input_df = input_df[
                categorical_features + numerical_features
            ]

            # Transform and predict
            processed_input = preprocessor.transform(input_df)

            predicted_price = model.predict(processed_input)[0]

            prediction = round(float(predicted_price), 2)

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        dropdowns=dropdowns,
        prediction=prediction,
        error=error,
        form_data=form_data
    )


# --------------------------------------------------
# 6. START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        use_reloader=False
    )
