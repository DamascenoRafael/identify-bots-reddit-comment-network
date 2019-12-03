import json
import requests
from tqdm import tqdm
from datetime import date

def collect_posts(username, since_date, post_handler):
    url = 'https://api.pushshift.io/reddit/search/submission/?author=' + username + '&fields=title&size=500'
    delta = date.today() - since_date
    
    for i in tqdm(range(delta.days)):
        before = '&before='+ str(i) + 'd'
        after = '&after=' + str(i+1) + 'd'
        new_url = url + before + after

        try:
            r = requests.get(new_url)
            if r.status_code != 200:
                print('RequestError:', new_url, 'status_code:', r.status_code)
                continue
            data = json.loads(r.content.decode())
            for post in data['data']:
                post_handler(post)
        except requests.exceptions.RequestException as e:
            print('RequestException:', e)
