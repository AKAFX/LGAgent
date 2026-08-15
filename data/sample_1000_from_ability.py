#!/usr/bin/env python3
"""
从 Ability_merged.jsonl 中随机抽取1000题，重新编号形成新的子数据集
"""

import json
import random
from pathlib import Path

def sample_and_renumber(input_file, output_file, sample_size=1000, seed=42):
    """
    从JSONL文件中随机抽取指定数量的记录，并重新编号
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        sample_size: 抽取数量
        seed: 随机种子（保证可复现）
    """
    print(f"正在读取: {input_file}")
    
    # 读取所有数据
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))
    
    total_count = len(all_data)
    print(f"总记录数: {total_count}")
    
    if sample_size > total_count:
        print(f"警告: 请求抽取 {sample_size} 条，但只有 {total_count} 条记录")
        sample_size = total_count
    
    # 设置随机种子
    random.seed(seed)
    
    # 随机抽取
    print(f"正在随机抽取 {sample_size} 条记录（随机种子: {seed}）...")
    sampled_data = random.sample(all_data, sample_size)
    
    # 重新编号
    print("正在重新编号...")
    for idx, item in enumerate(sampled_data):
        item['id'] = idx
        # 更新 meta_data 中的 original_index（保留原始索引信息）
        if 'meta_data' in item:
            if 'original_index' not in item['meta_data']:
                item['meta_data']['original_index'] = item.get('id', idx)
        else:
            item['meta_data'] = {'original_index': item.get('id', idx)}
    
    # 保存为新的JSONL文件
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 完成！")
    print(f"   输出文件: {output_file}")
    print(f"   抽取记录数: {len(sampled_data)}")
    print(f"   新编号范围: 0 - {len(sampled_data) - 1}")
    
    # 显示前3条示例
    print("\n前3条记录示例：")
    for i in range(min(3, len(sampled_data))):
        print(f"\n--- 记录 {i+1} (新ID: {sampled_data[i]['id']}) ---")
        print(f"问题: {sampled_data[i]['question'][:100]}...")
        print(f"答案: {sampled_data[i]['golden_answers']}")
        if 'meta_data' in sampled_data[i] and 'original_index' in sampled_data[i]['meta_data']:
            print(f"原始索引: {sampled_data[i]['meta_data']['original_index']}")

if __name__ == "__main__":
    # 输入文件
    input_file = Path("/ai/fx/data/fx/UltraRAG/data/Ability_merged.jsonl")
    
    # 输出文件
    output_file = Path("/ai/fx/data/fx/UltraRAG/data/Ability_sample_1000.jsonl")
    
    # 执行抽取和重新编号
    sample_and_renumber(input_file, output_file, sample_size=1000, seed=42)
