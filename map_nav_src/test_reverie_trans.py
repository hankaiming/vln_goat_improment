import json

# 明确指定输入和输出的绝对路径
input_path = '/workspace/VLN-GOAT/datasets/REVERIE/test/goat_reverie_valid_register1/preds/submit_val_unseen_dynamic.json'
output_path = '/workspace/VLN-GOAT/datasets/REVERIE/test/goat_reverie_valid_register1/preds/submit_val_unseen_dynamic1.json'

# 读取你的原始数据
with open(input_path, 'r') as f:
    data = json.load(f)

for item in data:
    # 1 & 2. 修改键名并将对象ID转为整数，同时处理值为 null/None 的情况
    if 'pred_objid' in item:
        val = item.pop('pred_objid')
        if val is not None:
            item['predObjId'] = int(val)
        else:
            # 如果模型输出是 null，这里保留为 None 或填入 -1，这里我们保留为空
            item['predObjId'] = None
    
    # 3. 补全 trajectory 格式，增加默认的 heading 和 elevation (0.0)
    if 'trajectory' in item:
        new_trajectory = []
        for step in item['trajectory']:
            if len(step) == 1:
                # 补充为 [viewpoint_id, heading_rads, elevation_rads]
                new_trajectory.append([step[0], 0.0, 0.0])
            else:
                new_trajectory.append(step)
        item['trajectory'] = new_trajectory

# 明确保存到指定的 preds/ 文件夹下
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"数据格式转换完成！文件已保存至：{output_path}")