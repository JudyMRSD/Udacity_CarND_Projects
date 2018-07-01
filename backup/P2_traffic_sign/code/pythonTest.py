import collections

y_train = [ 1, 3, 6, 10,3, 6, 3, 6, 10]

counter = collections.Counter(y_train)
print(counter)
print(counter.keys())
print(counter.values())
for k, v in counter.items():
    print (k, v)
