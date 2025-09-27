#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('.')
from evaluate_step_by_step_2 import LogicalReasoningEngine

def test_deduplication():
    """测试推理链去重功能"""
    
    engine = LogicalReasoningEngine()
    
    # 前提：包含复合条件
    premises = [
        {
            'input': 'qoslpus', 
            'output': 'babkpus and cugjpus', 
            'output_parsed': {'type': 'and', 'entities': ['babkpus', 'cugjpus']}, 
            'original': 'qoslpus is babkpus and cugjpus', 
            'type': 'premise'
        }
    ]
    
    # 目标：通过CE规则提取
    target = {
        'input': 'qoslpus', 
        'output': 'babkpus', 
        'output_parsed': {'type': 'single', 'entities': ['babkpus']}, 
        'original': 'qoslpus is babkpus', 
        'type': 'target'
    }
    
    print("=== 测试CE规则去重 ===")
    print(f"前提: {premises[0]['original']}")
    print(f"目标: {target['original']}")
    print(f"应该通过CE规则，只需要1个前提")
    
    print("\n=== 推理过程 ===")
    result = engine.is_provable(target, premises, debug=True, return_path=True)
    
    print(f"\n=== 结果 ===")
    print(f"可证明: {result['is_provable']}")
    
    if result['is_provable']:
        reasoning_depth = result.get('reasoning_depth', 0)
        print(f"\n📋 推理信息:")
        print(f"  推理深度: {reasoning_depth}")
        print(f"  使用前提: {len(result['reasoning_chain'])} 个")
        for i, step in enumerate(result['reasoning_chain']):
            print(f"    {i+1}. {step}")
        
        print(f"\n🔗 实体链: {' -> '.join(result['entities_chain'])}")
        
        print(f"\n✅ CE规则推理深度应该是1 (前提深度0 + 1层推理)")
        if reasoning_depth == 1:
            print(f"✅ 推理深度正确: {reasoning_depth}")
        else:
            print(f"❌ 推理深度错误: {reasoning_depth} (应该是1)")

if __name__ == "__main__":
    test_deduplication()
