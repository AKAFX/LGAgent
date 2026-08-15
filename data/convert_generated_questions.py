#!/usr/bin/env python3
"""
将 generated_questions.json 转换为评测标准格式（JSONL）
"""

import json
from pathlib import Path

def convert_to_standard_format(input_file, output_file):
    """
    将JSON数组格式转换为JSONL格式，并添加meta_data字段
    
    输入格式（JSON数组）：
    [
      {
        "id": 1,
        "question": "问题...",
        "golden_answers": ["A"]
      },
      ...
    ]
    
    输出格式（JSONL）：
    {"id": 0, "question": "问题...", "golden_answers": ["A"], "meta_data": {"correct_option": "A", "original_index": 1}}
    {"id": 1, "question": "问题...", "golden_answers": ["B"], "meta_data": {"correct_option": "B", "original_index": 2}}
    ...
    """
    print(f"正在读取: {input_file}")
    
    # 读取输入文件（JSON数组格式）
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"读取了 {len(data)} 条记录")
    print("正在转换格式...")
    
    # 转换格式
    converted_data = []
    for idx, item in enumerate(data):
        # 提取答案（golden_answers的第一个元素）
        golden_answers = item.get("golden_answers", [])
        correct_option = golden_answers[0] if golden_answers else ""
        
        # 构建输出格式
        output_item = {
            "id": idx,  # 重新编号，从0开始连续
            "question": item.get("question", "").strip(),
            "golden_answers": golden_answers,
            "meta_data": {
                "correct_option": correct_option,
                "original_index": item.get("id", idx)  # 保留原始id
            }
        }
        
        converted_data.append(output_item)
    
    # 保存为JSONL格式（每行一个JSON对象）
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 转换完成！")
    print(f"   输出文件: {output_file}")
    print(f"   总记录数: {len(converted_data)}")
    
    # 显示前3条示例
    print("\n前3条记录示例：")
    for i in range(min(3, len(converted_data))):
        print(f"\n--- 记录 {i+1} ---")
        print(json.dumps(converted_data[i], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 输入文件
    input_file = script_dir / "generated_questions.json"
    
    # 输出文件（JSONL格式）
    output_file = script_dir / "generated_questions.jsonl"
    
    # 执行转换
    convert_to_standard_format(input_file, output_file)
