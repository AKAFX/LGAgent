#!/usr/bin/env python3
"""
将合并后的JSON文件转换为项目评估所需格式
"""

import json
from pathlib import Path

def convert_to_eval_format(input_file, output_file):
    """
    将JSON格式转换为评估格式
    
    输入格式：
    {
      "question": "问题",
      "options": ["A. 选项1", "B. 选项2", ...],
      "answer": "A"
    }
    
    输出格式（JSONL）：
    {
      "id": 0,
      "question": "问题\n\nA. 选项1\nB. 选项2\n...",
      "golden_answers": ["A"],
      "meta_data": {}
    }
    """
    print(f"正在读取: {input_file}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"读取了 {len(data)} 条记录")
    print("正在转换格式...")
    
    # 转换格式
    converted_data = []
    for idx, item in enumerate(data):
        # 构建完整的问题（包含选项）
        question = item.get("question", "").strip()
        options = item.get("options", [])
        answer = item.get("answer", "").strip()
        
        # 将选项添加到问题后面
        if options:
            options_text = "\n".join(options)
            full_question = f"{question}\n\n{options_text}"
        else:
            full_question = question
        
        # 构建输出格式
        output_item = {
            "id": idx,
            "question": full_question,
            "golden_answers": [answer] if answer else [],
            "meta_data": {
                "correct_option": answer,
                "original_index": idx
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
    input_file = script_dir / "Ability_merged.json"
    
    # 输出文件（JSONL格式）
    output_file = script_dir / "Ability_merged.jsonl"
    
    # 执行转换
    convert_to_eval_format(input_file, output_file)
