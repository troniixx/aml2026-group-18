import sys
import os

signs = ['bird', 'boar', 'dog', 'dragon', 'hare', 
         'horse', 'monkey', 'ox', 'ram', 'rat', 
         'snake', 'tiger', 'zero']

path = 'data\\train\\'

for sign in signs:
    data = os.listdir(path+sign)
    for j in range(len(data)):
        old_path = path + sign + '\\' +  data[j]
        new_path = path + sign + '\\' + sign + '_' + str(j).rjust(4, '0')
        os.rename(old_path, new_path)