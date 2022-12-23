import requests
from bs4 import BeautifulSoup

response = requests.get('https://www.apimages.com/Collection/Landing/Arizona-Diamondbacks-2021-Team-Headshots/cb21659b3fce447c8f314dbe63f59e21/1')
soup = BeautifulSoup(response.content, 'html.parser')


# I want to extract all of the player page links from this link
# https://www.mlb.com/players
player_links = soup.find_all('img')
print(player_links)