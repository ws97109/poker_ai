import h5py
import json

def h5_to_dict(h5_file):
    def recursively_convert_h5_to_dict(h5_obj):
        result = {}
        for key, item in h5_obj.items():
            if isinstance(item, h5py.Dataset):  # 若是數據集
                result[key] = item[()].tolist()  # 轉換為列表格式
            elif isinstance(item, h5py.Group):  # 若是群組
                result[key] = recursively_convert_h5_to_dict(item)
        return result

    with h5py.File(h5_file, 'r') as f:
        return recursively_convert_h5_to_dict(f)

# 將 HDF5 轉換成 JSON
h5_file = '/Users/lishengfeng/Desktop/add ai poker game/model_9900epoch-1730200617.7844822.h5'
data_dict = h5_to_dict(h5_file)

# 儲存為 JSON 文件
json_file = 'model.json'
with open(json_file, 'w') as f:
    json.dump(data_dict, f, indent=4)