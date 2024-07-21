# nyc-airbnb-analysis
This project provides actionable insights for Airbnb hosts and business stakeholders in New York City. By optimizing pricing strategies, focusing on room types that attract more guests, and enhancing the overall guest experience, hosts can improve their performance on the Airbnb platform.


# NYC Airbnb Data Analysis

## Introduction
This project analyzes Airbnb listing data in New York City to uncover insights and trends that can inform business decisions. The analysis focuses on understanding various factors that influence Airbnb performance, including location, price, and customer reviews.

## Business Question
How can Airbnb hosts and business stakeholders optimize their listings and marketing strategies to improve occupancy rates and customer satisfaction in New York City?

## Dataset Description
The dataset used in this analysis is sourced from Kaggle and includes detailed information about Airbnb listings in New York City. It contains various attributes such as listing ID, host ID, neighborhood, room type, price, minimum nights, number of reviews, last review date, and availability.

## Data Cleaning and Preparation
Data cleaning steps included handling missing values, removing duplicates, and ensuring data consistency. Outliers were identified and treated to maintain the integrity of the analysis.

## Exploratory Data Analysis
Exploratory Data Analysis (EDA) was performed to understand the distribution of listings, prices, and reviews across different neighborhoods and room types. Key visualizations included:
- Distribution of listings by neighborhood group
- Price distribution across different neighborhoods
- Review patterns over time

## Key Findings
- **Price Trends:** Listings in Manhattan are generally more expensive than those in other boroughs, with a significant price variation within the borough itself.
- **Occupancy Rates:** Private rooms have higher occupancy rates compared to entire home/apartment listings.
- **Review Insights:** Listings with higher review scores tend to have more consistent occupancy rates.

## Recommendations
- **Pricing Strategy:** Hosts in Manhattan can consider competitive pricing strategies to attract more bookings, especially in high-demand areas.
- **Room Type Optimization:** Offering private rooms can increase occupancy rates, particularly for budget-conscious travelers.
- **Enhancing Reviews:** Improving guest experience to garner better reviews can positively impact occupancy and pricing power.

## Usage
To run this analysis, clone the repository and open the Jupyter notebook using JupyterLab or Jupyter Notebook. The analysis is performed using Python libraries including pandas, numpy, seaborn, and matplotlib.

```bash
git clone https://github.com/Ranas1ali/nyc-airbnb-analysis.git
cd nyc-airbnb-analysis
jupyter notebook nyc-airbnb-analysis.ipynb
