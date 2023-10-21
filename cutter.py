import os
import csv

path = "E://studia//magisterka//magisterka//Mouse-Dynamics-Challenge-master-2//test_files"
users = (os.listdir(path))


def checkSession(folder):
    is_there = 0
    input_file  = open('Mouse-Dynamics-Challenge-master-2\public_labels.csv', "r")
    reader = csv.DictReader(input_file)
    for line in reader:
        file = line['filename']
        is_illegal = line['is_illegal']
        if folder in file:
            is_there = 1
            # print(folder, file, is_there)
    if is_there == 0:
        print("ITS NOT THERE")
for user in users:
    new_path = os.listdir(os.path.join(path, user))
    for folder in new_path:
        checkSession(folder)

