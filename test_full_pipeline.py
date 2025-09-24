#!/usr/bin/env python3
"""
测试evaluate_step_by_step_2的完整链路
包括：LLM提取形式化节点 → 后处理验证
"""

import json
from evaluate_step_by_step_2 import StepByStepEvaluator2

def test_full_pipeline():
    """测试完整的评估链路"""
    
    print("=== 测试evaluate_step_by_step_2完整链路 ===")
    
    # 从evaluation_results文件中加载一个例子
    with open('./evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json', 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 选择第一个例子
    example = eval_data['details'][0]
    
    # 从LoG_5.jsonl获取对应的graph数据
    with open('./generated_data/LoG_5.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        log_example = json.loads(lines[example['index']])
        example['graph'] = log_example.get('graph', [])
    
    print(f"选择例子: Index {example['index']}")
    print(f"预期答案: {example.get('expected', 'N/A')}")
    print(f"模型预测: {example.get('predicted', 'N/A')}")
    print(f"状态: {example.get('status', 'N/A')}")
    print(f"LoG图节点数: {len(example.get('graph', []))}")
    
    # 创建评估器 - 使用LLM调试模式（会调用LLM但跳过一些后处理）
    evaluator = StepByStepEvaluator2(
        api_key="sk-b56f448069294b79967b8c897aebcec3",  # 使用真实API key
        model_name="deepseek-reasoner",
        debug_mode=False,  # 启用API调用
        llm_debug_mode=False  # 启用完整后处理
    )
    
    print(f"\n=== 开始完整评估流程 ===")
    
    # 创建临时日志文件
    temp_log = {
        "total": 1,
        "details": [example]
    }
    
    temp_log_path = "temp_test_log.json"
    with open(temp_log_path, 'w', encoding='utf-8') as f:
        json.dump(temp_log, f, ensure_ascii=False, indent=2)
    
    try:
        # 运行完整的评估流程
        results = evaluator.evaluate_reasoning_process(
            log_path=temp_log_path,
            output_path="temp_test_output.json"
        )
        
        print(f"\n=== 评估完成 ===")
        print(f"处理的记录数: {results.get('total_records', 0)}")
        
        # 检查处理结果
        processed_records = results.get('processed_records', [])
        if processed_records:
            record = processed_records[0]
            
            print(f"\n=== Statement列表结果 ===")
            statement_list = record.get('statement_list', [])
            print(f"Statement节点数: {len(statement_list)}")
            
            for i, stmt in enumerate(statement_list):
                print(f"  {i+1:2d}. [{stmt['node_type']:12s}] {stmt['original_statement']}")
                print(f"      出现次数: {stmt['occurrence_count']}, 正确性: {stmt['is_correct']}")
            
            print(f"\n点亮的LoG节点数: {len(record.get('illuminated_log_nodes', []))}")
            for node in record.get('illuminated_log_nodes', []):
                print(f"  - {node}")
        
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        import os
        try:
            os.remove(temp_log_path)
            os.remove("temp_test_output.json")
        except:
            pass

if __name__ == "__main__":
    test_full_pipeline()
