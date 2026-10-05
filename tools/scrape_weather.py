import requests
from bs4 import BeautifulSoup
from csv import writer

url = "https://www.wunderground.com/hourly/us/mi/houghton"

response = requests.get(url)

soup = BeautifulSoup(response.text, "html.parser")

# td is the label for table data in the html. this is an array of all html content that is considered table data.
cells = soup.select("td")

# sometimes scraping the html returns nothing. this makes sure that it is fetched repeatedly until something is returned.
while len(cells) == 0:
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    cells = soup.select("td")

time = cells[0].get_text()
conditions = cells[1].get_text()
temp = cells[2].get_text()
precip = cells[4].get_text()
inches = cells[5].get_text()
humidity = cells[8].get_text()
wind = cells[9].get_text()

row = [time, conditions, temp, precip, inches, humidity, wind]

# add row of data to weather_data.csv
with open("data/weather_data.csv", "a", newline="") as f_object:
    writer_object = writer(f_object)
    writer_object.writerow(row)
    f_object.close()

# print(time)
# print(conditions)
# print(temp)
# print(precip)
# print(inches)
# print(humidity)
# print(wind)