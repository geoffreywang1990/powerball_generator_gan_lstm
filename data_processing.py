import numpy as np
import pandas as pd

import requests
from bs4 import BeautifulSoup
def fetch_powerball_numbers(begin_year=2022, end_year=2023):
    numbers = [] 

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for year in range(begin_year, end_year+1):
        url = f"https://powerfall.com/PastWinningLotteryNumbers/powerball/{year}/"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        table_rows = soup.select('table tbody tr')

        for row in table_rows:
            columns = row.select('td')
            if len(columns) > 1:
                date = columns[0].get_text()
                number_set = columns[1].get_text()
                numbers.append({'date': date, 'numberSet': number_set})

    print(f"获得一共{len(numbers)}条数据")
    return numbers



def preprocess_data(numbers):
    # Convert numbers list to DataFrame
    df = pd.DataFrame(numbers)

    # Convert date to datetime format and sort by date
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    # Process numberSet
    number_sets = df['numberSet'].str.split(',')
    df['Blue Balls'] = number_sets.apply(lambda x: [int(i) for i in x[:-1]])
    df['Red Ball'] = number_sets.apply(lambda x: int(x[-1]))

    return df

def data_processing(begin_year=2020,end_year=2023):
    numbers = fetch_powerball_numbers(begin_year,end_year)
    return preprocess_data(numbers)

if __name__ == '__main__':
    print(data_processing())
