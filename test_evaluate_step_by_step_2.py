#!/usr/bin/env python3
"""
测试evaluate_step_by_step_2.py的修改功能
"""

import sys
import os

def test_single_record_debug_mode():
    """测试单条记录调试模式"""
    print("测试单条记录调试模式...")
    
    # 构造测试命令
    cmd = [
        "python", "evaluate_step_by_step_2.py",
        "--log_path", "evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json",
        "--debug_mode",  # 跳过API调用
        "--single_record_debug",  # 只处理第一条记录
        "--output_path", "test_single_record_output.json"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 运行测试
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(f"返回码: {result.returncode}")
        print(f"标准输出: {result.stdout}")
        if result.stderr:
            print(f"标准错误: {result.stderr}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("测试超时")
        return False
    except Exception as e:
        print(f"测试出错: {e}")
        return False

def test_all_records_mode():
    """测试处理所有记录模式"""
    print("测试处理所有记录模式...")
    
    # 构造测试命令
    cmd = [
        "python", "evaluate_step_by_step_2.py",
        "--log_path", "evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json",
        "--debug_mode",  # 跳过API调用
        "--output_path", "test_all_records_output.json"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 运行测试
    import subprocess
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 更长的超时时间
        print(f"返回码: {result.returncode}")
        print(f"标准输出: {result.stdout}")
        if result.stderr:
            print(f"标准错误: {result.stderr}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("测试超时")
        return False
    except Exception as e:
        print(f"测试出错: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试evaluate_step_by_step_2.py的修改...")
    
    # 检查必要的文件是否存在
    required_files = [
        "evaluate_step_by_step_2.py",
        "evaluation_log/evaluation_results.LoG_5.deepseek-reasoner.json",
        "extract_prompt_2.txt"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"错误: 找不到必要文件: {file_path}")
            return 1
    
    # 测试单条记录调试模式
    print("\n" + "="*60)
    test1_success = test_single_record_debug_mode()
    print(f"单条记录调试模式测试: {'成功' if test1_success else '失败'}")
    
    # 测试处理所有记录模式
    print("\n" + "="*60)
    test2_success = test_all_records_mode()
    print(f"处理所有记录模式测试: {'成功' if test2_success else '失败'}")
    
    # 总结
    print("\n" + "="*60)
    print("测试总结:")
    print(f"  单条记录调试模式: {'✓' if test1_success else '✗'}")
    print(f"  处理所有记录模式: {'✓' if test2_success else '✗'}")
    
    if test1_success and test2_success:
        print("所有测试通过！")
        return 0
    else:
        print("部分测试失败，请检查代码。")
        return 1

if __name__ == "__main__":
    exit(main())
