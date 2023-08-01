import os

directory = 'data'
master = []

for filename in os.listdir(directory):
    if filename[-3:] == 'jpg':
        master.append(filename)

print(master)

ratio = 0.8

train_list = master[0:int(len(master)*ratio)]
test_list = master[int(len(master)*ratio):]

with open('rus/train.gc', 'w') as f:
    for line in train_list:
        f.write(line)
        f.write('\n')
    f.close()

with open('rus/test.gc', 'w') as f:
    for line in test_list:
        f.write(line)
        f.write('\n')
    f.close()

