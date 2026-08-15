#!/usr/bin/env python3
"""
合并 mdata 文件夹中的20个JSON文件为一个文件
"""

import json
import os
from pathlib import Path

def merge_json_files(input_dir, output_file):
    """
    合并指定目录中的所有JSON文件
    
    Args:
        input_dir: 输入目录路径
        output_file: 输出文件路径
    """
    # 获取所有JSON文件，按数字顺序排序
    input_path = Path(input_dir)
    json_files = sorted(
        [f for f in input_path.glob("Ability_*.json")],
        key=lambda x: int(x.stem.split('_')[1])  # 按数字排序
    )
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 合并所有数据
    merged_data = []
    total_items = 0
    
    for json_file in json_files:
        print(f"正在处理: {json_file.name}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                    total_items += len(data)
                    print(f"  - 添加了 {len(data)} 条记录")
                else:
                    print(f"  - 警告: {json_file.name} 不是数组格式，跳过")
        except json.JSONDecodeError as e:
            print(f"  - 错误: {json_file.name} JSON解析失败: {e}")
        except Exception as e:
            print(f"  - 错误: {json_file.name} 读取失败: {e}")
    
    # 保存合并后的数据
    print(f"\n总共合并了 {total_items} 条记录")
    print(f"正在保存到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 合并完成！输出文件: {output_file}")
    print(f"   总记录数: {len(merged_data)}")

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 输入目录（当前目录）
    input_directory = script_dir
    
    # 输出文件
    output_filename = script_dir / "Ability_merged.json"
    
    # 执行合并
    merge_json_files(input_directory, output_filename)
