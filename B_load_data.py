from A_libraries import*

# luu data vao train_files, gom train va mask
train_files = []
mask_files = glob('kaggle_3m/*/*_mask*')

for i in mask_files:
    train_files.append(i.replace('_mask', ''))
    
print(train_files[:10])
print(mask_files[:10])