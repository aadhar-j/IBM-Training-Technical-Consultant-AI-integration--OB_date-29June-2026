import requests


def weather_tool(city: str):

    """
    Returns current weather of a city using
    OpenStreetMap + Open-Meteo APIs.
    """

    # -----------------------------
    # Step 1 : Get Latitude & Longitude
    # -----------------------------

    geo_url = (
        f"https://nominatim.openstreetmap.org/search"
        f"?q={city}&format=json&limit=1"
    )

    geo_response = requests.get(
        geo_url,
        headers={
            "User-Agent": "LangGraph-RAG-Demo"
        }
    ).json()

    if len(geo_response) == 0:
        return f"City '{city}' not found."

    latitude = geo_response[0]["lat"]
    longitude = geo_response[0]["lon"]

    # -----------------------------
    # Step 2 : Get Weather
    # -----------------------------

    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        f"&current=temperature_2m,"
        f"relative_humidity_2m,"
        f"wind_speed_10m"
    )

    weather = requests.get(weather_url).json()

    current = weather["current"]

    result = f"""
City : {city}

Temperature : {current['temperature_2m']} °C

Humidity : {current['relative_humidity_2m']} %

Wind Speed : {current['wind_speed_10m']} km/h
"""

    return result