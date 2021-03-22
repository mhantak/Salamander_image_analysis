#Testing getDatasets() with cross validation implementation 

from Cross_valid_salamander_data import getDatasets

#train_data, val_data = getDatasets('/blue/guralnick/mhantak/categories_binary/')

train, test = getDatasets(path='/blue/guralnick/mhantak/categories_binary/', train_size=0.5) #, 0.5
print(len(train), len(test))

train_idx = list(range(2500))
valid_idx = list(range(2500, 3871)) #1371

train, test = getDatasets(path='/blue/guralnick/mhantak/categories_binary/', train_idx=train_idx, valid_idx=valid_idx)
print(len(train), len(test))