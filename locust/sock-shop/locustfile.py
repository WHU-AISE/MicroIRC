#!/usr/bin/python3

from random import randint, choice
import base64
from locust import HttpUser, task, between
from locust.clients import HttpSession
import locust.stats

locust.stats.CSV_STATS_INTERVAL_SEC = 5 # default is 1 second
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 5 # Determines how often the data is flushed to disk, default is 10 seconds

def get_base64(username, password):
  string = "%s:%s" % (username, password)
  string = string.encode()
  base64string = base64.b64encode(string)
  return base64string

auth_header = get_base64("a", "a").decode()
print(auth_header)

class httpUser(HttpUser):
  wait_time = between(0.1, 0.5)

  @task
  def load(self):
    catalogue = self.client.get("/catalogue").json()
    category_item = choice(catalogue)
    item_id = category_item["id"]
    self.client.get("/")
    self.client.get("/login", headers={"Authorization":"Basic %s" % auth_header})
    self.client.get("/customers")
    self.client.get("/category.html")
    self.client.get("/detail.html?id={}".format(item_id))
    self.client.delete("/cart")
    self.client.post("/cart", json={"id": item_id, "quantity": 1})
    self.client.get("/basket.html")
    self.client.post("/orders")
