import redis

client = redis.Redis()

with open(r'C:\Users\jeffk\Documents\Work\tensorflow-inference-master\src\main\resources\fm\model.pb', 'rb') as f:
    model = f.read()

client.set("fm_model", model, 5)

# print(client.get("fm_model"))
