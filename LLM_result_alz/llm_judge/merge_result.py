import json

combined_list = []
input_files = ['judge_results_shard0.jsonl', 'judge_results_shard1.jsonl']

for file_name in input_files:
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 确保跳过空行
                # json.loads 作用于文件中的“每一行”
                data = json.loads(line)
                combined_list.append(data)

# 现在 combined_list 包含了两个文件中的所有数据
print(f"总共合并了 {len(combined_list)} 条数据")

# save result as file
output_file = 'combined_judge_results.jsonl'

# 使用 'w' 模式打开文件（会创建或覆盖文件）
with open(output_file, 'w', encoding='utf-8') as f:
    for item in combined_list:
        # ensure_ascii=False 可以保证中文不乱码
        # + '\n' 确保每条数据占独立的一行，符合 JSONL 格式
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"数据已成功保存到：{output_file}")