from A_libraries import*
from B_load_data import*
from C_check_data import*
# MOve mask files and train files to df, then I share data -> (train, test)
df = pd.DataFrame(data={'filename': train_files, 'mask': mask_files})
df_train, df_test = train_test_split(df, test_size= 0.1)
df_train, df_val = train_test_split(df_train, test_size= 0.2)

print(df_train.values.shape)
print(df_test.values.shape)
print(df_val.values.shape)
