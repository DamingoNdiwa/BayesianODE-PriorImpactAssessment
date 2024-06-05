import requests
import pandas as pd


def download_data(url):
    """To get data from the internet

    Args:
        url (_type_): link to the website
    """
    response = requests.get(url)
    if response.status_code == 200:
        with open("covid_data.csv", "wb") as file:
            file.write(response.content)
        print("Data downloaded successfully.")
    else:
        print("Error downloading data. Status code:", response.status_code)


# To get data from the web
# Acesses in january 2023
url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
# URL of the COVID-19 data source
# download_data(url)# Uncomment to call function


df = pd.read_csv("covid_data.csv")

# Select data for first Wave inluxembourg
filtered_df = df.loc[
    (df["date"] >= "2020-02-29")
    & (df["date"] <= "2020-06-13")
    & (df["location"] == "Luxembourg"),
    [
        "date",
        "population",
        "total_cases",
        "new_cases",
        "hosp_patients",
        "icu_patients",
        "reproduction_rate",
        "total_deaths",
        "new_deaths",
    ],
]
data_plot = df.loc[
    (df["date"] >= "2020-02-29")
    & (df["date"] <= "2020-07-29")
    & (df["location"] == "Luxembourg"),
    [
        "date",
        "population",
        "total_cases",
        "new_cases",
        "hosp_patients",
        "icu_patients",
        "reproduction_rate",
        "total_deaths",
        "new_deaths",
    ],
]

# Print the column names of the filtered dataframe
filtered_df.columns.tolist()

# save filtered_df as csv file
filtered_df.to_csv("filtered_data.csv", index=False)
