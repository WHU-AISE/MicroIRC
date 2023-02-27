#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
import locust.stats
from locust import HttpUser, between, task

locust.stats.CSV_STATS_INTERVAL_SEC = 5 # default is paymentservice_timeout second
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 5 # Determines how often the data is flushed to disk, default is 10 seconds

products = [
'0PUK6V6EV0',
'1YMWWN1N4O',
'2ZYFJ3GM2N',
'66VCHSJNUP',
'6E92ZMYYFZ',
'9SIQT8TOJO',
'L9ECAV7KIM',
'LS4PSXUNUM',
'OLJCESPC7Z']


class WebsiteUser(HttpUser):
    wait_time = between(1, 10)

    @task(2)
    def index(self):
        self.client.get("/")

    @task(2)
    def setCurrency(self):
        currencies = ['EUR', 'USD', 'JPY', 'CAD', 'GBP', 'TRY']
        self.client.post("/setCurrency",
            {'currency_code': random.choice(currencies)})

    @task(10)
    def browseProduct(self):
        self.client.get("/product/" + random.choice(products))

    @task(3)
    def viewCart(self):
        self.client.get("/cart")

    @task(2)
    def addToCart(self):
        product = random.choice(products)
        self.client.get("/product/" + product)
        self.client.post("/cart", {
            'product_id': product,
            'quantity': random.choice([1,2,3,4,5,10])})

    @task(2)
    def checkout(self):
        # addToCart(self)
        self.client.post("/cart/checkout", {
            'email': 'someone@example.com',
            'street_address': '1600 Amphitheatre Parkway',
            'zip_code': '94043',
            'city': 'Mountain View',
            'state': 'CA',
            'country': 'United States',
            'credit_card_number': '4432-8015-6152-0454',
            'credit_card_expiration_month': 'paymentservice_timeout',
            'credit_card_expiration_year': '2022',
            'credit_card_cvv': '672',
        })


