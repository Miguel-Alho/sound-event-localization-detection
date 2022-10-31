import os

PATH = ''

for folder in os.listdir(PATH):
    index = folder.find(',0)_') 
    if index != -1:
        old_path = os.path.join(PATH, folder)
        new_path = os.path.join(PATH, folder).replace(',0)_', ')_')
        os.rename(old_path, new_path)

