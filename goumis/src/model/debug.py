import os

data_dir = "./src/model/greentext_data"
if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"Files in {data_dir}:")
    for f in files:
        print(f"  - {f}")
else:
    print(f"Directory {data_dir} does not exist!")