import dataset

lines = dataset.readLines("/var/wqs/weibo_dialogue/relations")
lines = [x.split(":", 1) for x in lines]
groups = [(int(x[0]), [y for y in x[1].split(",")]) for x in lines]
pairs = []
for g in groups:
    for x in g[1]:
        pairs.append((g[0], x))

for p in pairs:
    print("{} {} f".format(p[0], p[1]))
