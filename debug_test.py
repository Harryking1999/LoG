def parse_output_entities(output_part):
    """解析output部分的实体结构"""
    output_part = output_part.strip()
    
    if " and " in output_part:
        entities = [e.strip() for e in output_part.split(" and ")]
        return {"type": "and", "entities": entities}
    elif " or " in output_part:
        entities = [e.strip() for e in output_part.split(" or ")]
        return {"type": "or", "entities": entities}
    else:
        return {"type": "single", "entities": [output_part]}

def parse_statement(statement_str, input_part, output_part):
    """解析语句字符串，构造标准格式的字典"""
    output_parsed = parse_output_entities(output_part)
    
    return {
        "original": statement_str,
        "input": input_part,
        "output": output_part,
        "output_parsed": output_parsed,
        "type": "actual"
    }

def statements_equal(stmt1, stmt2):
    """判断两个语句是否相等"""
    return (stmt1["input"] == stmt2["input"] and 
            stmt1["output"] == stmt2["output"])

def output_contains_entity(output_parsed, target_entity):
    """检查output是否包含目标实体"""
    return target_entity in output_parsed["entities"]

def is_provable(target, premises, visited=None, depth=0, max_depth=2000, debug=False, start_time=None, timeout=120):
    """
    反向推理：判断目标是否可以从前提中推导出来
    
    Args:
        target: 目标结论 {"input": "x", "output": "babcpus", "output_parsed": {...}}
        premises: 前提条件列表
        visited: 已访问的目标集合，防止循环
        depth: 递归深度
        max_depth: 最大递归深度
        debug: 是否打印调试信息
    """
    if visited is None:
        visited = set()
    
    if start_time is None:
        start_time = time.time()
    
    # 超时检查
    if time.time() - start_time > timeout:
        if debug:
            print(f"推理超时({timeout}s)，终止")
        return False
    
    indent = "  " * depth
    if debug:
        print(f"{indent}尝试证明: {target['input']} is {target['output']}")
    
    # 简化循环检测：只防止直接无限递归，不限制深度
    target_key = f"{target['input']}→{target['output']}"
    
    # 如果在当前路径中已经访问过这个目标，直接跳过（防止A→B→A的直接循环）
    if target_key in visited:
        if debug:
            print(f"{indent}检测到循环依赖，跳过")
        return False
    
    if depth > max_depth:
        if debug:
            print(f"{indent}超过最大深度({max_depth})，跳过")
        return False
    
    # 临时添加到visited，但在尝试路径时会传递副本
    if target_key not in visited:
        visited.add(target_key)
    
    try:
        # 基础情况：目标已经在前提中
        for premise in premises:
            if statements_equal(target, premise):
                if debug:
                    print(f"{indent}✓ 在前提中找到: {premise['original']}")
                return True
        
        # 寻找可能的推理路径
        possible_paths = find_reasoning_paths(target, premises, debug and depth < 5)
        
        if debug:
            print(f"{indent}找到 {len(possible_paths)} 种推理路径")
        
        # 按优先级排序路径：优先尝试简单的路径，并优先考虑直接匹配
        def path_priority(path):
            # CE和DI_EXPAND路径通常更直接
            rule_priority = {
                'CE': 1,
                'DI_EXPAND': 2, 
                'DI': 3,
                'MP': 4,
                'CI': 5,
                'MP+CE': 6
            }
            return (len(path['intermediates']), rule_priority.get(path['rule'], 10))
        
        possible_paths.sort(key=path_priority)
        
        # 尝试所有推理路径，只受超时限制
        # 移除路径数量限制，让算法有机会找到所有可能的解
        
        for i, path in enumerate(possible_paths):
            # 超时检查
            if time.time() - start_time > timeout:
                if debug:
                    print(f"{indent}推理超时，停止尝试更多路径")
                break
                
            if debug:
                print(f"{indent}尝试路径 {i+1}: {path['rule']}")
                for j, intermediate in enumerate(path['intermediates']):
                    print(f"{indent}  需要: {intermediate['input']} is {intermediate['output']}")
            
            # 检查这条路径的所有中间步骤是否都可以证明
            all_provable = True
            for intermediate in path['intermediates']:
                # 使用当前visited的副本，避免影响其他路径
                if not is_provable(intermediate, premises, visited.copy(), depth + 1, max_depth, debug, start_time, timeout):
                    all_provable = False
                    break
            
            if all_provable:
                if debug:
                    print(f"{indent}✓ 路径 {i+1} 成功")
                return True
            elif debug:
                print(f"{indent}✗ 路径 {i+1} 失败")
        
        if debug:
            print(f"{indent}✗ 所有路径都失败")
        return False
        
    finally:
        # 移除当前目标的访问记录，允许其他路径访问
        visited.discard(target_key)

def find_reasoning_paths(target, premises, debug=False):
    """
    从已有前提中寻找可以推导出目标的推理路径
    
    Returns:
        List[Dict]: 每个元素包含 {"rule": "规则名", "intermediates": [中间步骤列表]}
    """
    paths = []
    target_input = target["input"]
    target_output = target["output"]
    target_output_parsed = target["output_parsed"]
    
    # 规则1: MP (Modus Ponens)
    # 要证明 x is babcpus，寻找：
    # 1) x is A (其中A是某个中间值)
    # 2) A is babcpus 或 A is ... and babcpus ... 等包含babcpus的条件
    mp_paths = find_mp_paths(target, premises, debug)
    paths.extend(mp_paths)
    
    # 规则2: CE (Conjunction Elimination) 
    # 要证明 x is babcpus，寻找 x is babcpus and ... 的条件
    ce_paths = find_ce_paths(target, premises, debug)
    paths.extend(ce_paths)
    
    # 规则3: CI (Conjunction Introduction)
    # 要证明 x is A and B，需要 x is A 和 x is B
    ci_paths = find_ci_paths(target, premises, debug)
    paths.extend(ci_paths)
    
    # 规则4: DI (Disjunction Introduction)
    # 要证明 x is A or B，只需要 x is A 或 x is B 中的一个
    di_paths = find_di_paths(target, premises, debug)
    paths.extend(di_paths)
    
    return paths

def find_mp_paths(target, premises, debug=False):
    """寻找MP规则的推理路径"""
    paths = []
    target_input = target["input"]
    target_output = target["output"]
    target_output_parsed = target["output_parsed"]
    
    # 寻找形如 target_input is X 的前提
    for premise in premises:
        if premise["input"] == target_input:
            # 找到了 target_input is X
            x_value = premise["output"]
            x_parsed = premise["output_parsed"]
            
            # 情况1: 如果X是单个值，寻找 X is target_output
            if x_parsed["type"] == "single":
                intermediate_target = {
                    "input": x_value,
                    "output": target_output,
                    "output_parsed": target_output_parsed
                }
                paths.append({
                    "rule": "MP",
                    "intermediates": [intermediate_target]
                })
                
                if debug:
                    print(f"    MP路径: {target_input} is {x_value} → 需要证明 {x_value} is {target_output}")
            
            # 情况2: 如果X是复合值(如A and B)，可以通过CE提取单个部分，然后继续MP
            elif x_parsed["type"] == "and":
                for entity in x_parsed["entities"]:
                    # 先通过CE得到 target_input is entity，再通过MP得到最终目标
                    ce_intermediate = {
                        "input": target_input,
                        "output": entity,
                        "output_parsed": {"type": "single", "entities": [entity]}
                    }
                    mp_intermediate = {
                        "input": entity,
                        "output": target_output,
                        "output_parsed": target_output_parsed
                    }
                    
                    paths.append({
                        "rule": "MP+CE",
                        "intermediates": [ce_intermediate, mp_intermediate]
                    })
                    
                    if debug:
                        print(f"    MP+CE路径: {target_input} is {x_value} → CE得到 {target_input} is {entity} → MP需要 {entity} is {target_output}")
    
    return paths

def find_ce_paths(target, premises, debug=False):
    """寻找CE规则的推理路径"""
    paths = []
    target_input = target["input"]
    target_output = target["output"]
    target_output_parsed = target["output_parsed"]
    
    # 如果目标是单个实体，寻找包含它的复合条件
    if target_output_parsed["type"] == "single":
        target_entity = target_output_parsed["entities"][0]
        
        # 寻找形如 target_input is ... and target_entity ... 的条件
        for premise in premises:
            if (premise["input"] == target_input and 
                premise["output_parsed"]["type"] == "and" and
                target_entity in premise["output_parsed"]["entities"]):
                
                # 找到了可以通过CE得到目标的复合条件
                intermediate = {
                    "input": target_input,
                    "output": premise["output"],
                    "output_parsed": premise["output_parsed"]
                }
                paths.append({
                    "rule": "CE",
                    "intermediates": [intermediate]
                })
                
                if debug:
                    print(f"    CE路径: {premise['output']} contains {target_output}")
    
    return paths

def find_ci_paths(target, premises, debug=False):
    """寻找CI规则的推理路径"""
    paths = []
    target_input = target["input"]
    target_output_parsed = target["output_parsed"]
    
    # 只对and类型的目标应用CI规则
    if target_output_parsed["type"] == "and":
        entities = target_output_parsed["entities"]
        
        # 需要为每个实体找到单独的条件
        intermediates = []
        for entity in entities:
            intermediate = {
                "input": target_input,
                "output": entity,
                "output_parsed": {"type": "single", "entities": [entity]}
            }
            intermediates.append(intermediate)
        
        paths.append({
            "rule": "CI",
            "intermediates": intermediates
        })
        
        if debug:
            print(f"    CI路径: 需要 {len(entities)} 个单独的条件")
    
    return paths

def find_di_paths(target, premises, debug=False):
    """寻找DI规则的推理路径"""
    paths = []
    target_input = target["input"]
    target_output_parsed = target["output_parsed"]
    
    # 只对or类型的目标应用DI规则
    if target_output_parsed["type"] == "or":
        target_entities = set(target_output_parsed["entities"])
        
        # 方法1: 为每个实体创建一个单独的路径（只需要其中一个成功）
        for entity in target_entities:
            intermediate = {
                "input": target_input,
                "output": entity,
                "output_parsed": {"type": "single", "entities": [entity]}
            }
            
            paths.append({
                "rule": "DI",
                "intermediates": [intermediate]
            })
            
            if debug:
                print(f"    DI路径: 只需要证明 {target_input} is {entity}")
        
        # 方法2: 寻找包含目标实体子集的or条件（DI扩展）
        # 如果目标是 A or B or C，可以从 A or B 推导出来
        for premise in premises:
            if (premise["input"] == target_input and 
                premise["output_parsed"]["type"] == "or"):
                premise_entities = set(premise["output_parsed"]["entities"])
                
                # 如果前提的实体是目标实体的子集，可以通过DI扩展
                if premise_entities.issubset(target_entities) and premise_entities != target_entities:
                    intermediate = {
                        "input": target_input,
                        "output": premise["output"],
                        "output_parsed": premise["output_parsed"]
                    }
                    
                    paths.append({
                        "rule": "DI_EXPAND",
                        "intermediates": [intermediate]
                    })
                    
                    if debug:
                        print(f"    DI扩展路径: 从 {target_input} is {premise['output']} 扩展到 {target['output']}")
    
    return paths

def can_derive_output(source_parsed, target_parsed):
    """检查是否可以从source_parsed推导出target_parsed"""
    # 如果目标是单个实体
    if target_parsed["type"] == "single":
        target_entity = target_parsed["entities"][0]
        return target_entity in source_parsed["entities"]
    
    # 如果目标是and组合，source必须包含所有实体
    elif target_parsed["type"] == "and":
        target_entities = set(target_parsed["entities"])
        source_entities = set(source_parsed["entities"])
        return target_entities.issubset(source_entities)
    
    # 如果目标是or组合，source包含任一实体即可
    elif target_parsed["type"] == "or":
        target_entities = set(target_parsed["entities"])
        source_entities = set(source_parsed["entities"])
        return len(target_entities.intersection(source_entities)) > 0
    
    return False

import json
import re
import time

def load_log_data(file_path):
    """加载LoG数据文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def extract_given_conditions(question_text):
    """从问题文本中提取Given Information"""
    # 提取Given Information部分
    given_match = re.search(r'\*\*Given Information\*\*:\s*(.*?)\s*\*\*', 
                           question_text, re.DOTALL)
    
    conditions = []
    if given_match:
        given_text = given_match.group(1).strip()
        # 按句号分割条件
        raw_conditions = re.split(r'\.\s*', given_text)
        
        for condition in raw_conditions:
            condition = condition.strip()
            if condition and ' is ' in condition:
                conditions.append(condition)
    
    return conditions

def extract_target_question(question_text):
    """从问题文本中提取目标问题"""
    # 提取Question部分
    question_match = re.search(r'\*\*Question\*\*:\s*Is it true or false or unkown:\s*(.*?)\?', 
                              question_text, re.DOTALL)
    
    if question_match:
        return question_match.group(1).strip()
    
    return None

def test_log_examples(file_path='./generated_data/LoG_5.jsonl', test_count=30):
    """测试LoG数据文件中的例子"""
    
    # 加载数据
    data = load_log_data(file_path)
    print(f"加载了 {len(data)} 个例子 from {file_path}")
    
    # 测试前test_count个例子
    correct_count = 0
    total_count = min(test_count, len(data))
    
    for i in range(total_count):
        example = data[i]
        print(f"\n=== 测试例子 {i+1} (ID: {example['id']}) ===")
        
        question = example['question']
        expected_answer = example['answer']
        
        # 提取条件和目标
        conditions = extract_given_conditions(question)
        target_question = extract_target_question(question)
        
        print(f"预期答案: {expected_answer}")
        print(f"目标问题: {target_question}")
        print(f"条件数量: {len(conditions)}")
        
        # 解析条件为前提
        premises = []
        for condition in conditions:
            if ' is ' in condition:
                parts = condition.split(' is ', 1)
                if len(parts) == 2:
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    premises.append(parse_statement(condition, input_part, output_part))
        
        # 解析目标问题
        if target_question and ' is ' in target_question:
            parts = target_question.split(' is ', 1)
            if len(parts) == 2:
                input_part = parts[0].strip()
                output_part = parts[1].strip()
                target = parse_statement(target_question, input_part, output_part)
                
                # 对于ID=17的例子，打印详细信息
                if example['id'] == 17:
                    print("详细条件:")
                    for j, condition in enumerate(conditions):
                        print(f"  {j+1}. {condition}")
                    
                    print("\n解析后的前提:")
                    for j, premise in enumerate(premises):
                        print(f"  {j+1}. {premise['input']} is {premise['output']} (type: {premise['output_parsed']['type']})")
                        if premise['output_parsed']['type'] != 'single':
                            print(f"      entities: {premise['output_parsed']['entities']}")
                    
                    print(f"\n目标解析: {target}")
                    print(f"目标类型: {target['output_parsed']['type']}")
                    print(f"目标实体: {target['output_parsed']['entities']}")
                    
                    # 特别检查关键前提
                    for premise in premises:
                        if 'babcpus' in premise['output'] or 'babrpus' in premise['output']:
                            print(f"\n关键前提: {premise['input']} is {premise['output']}")
                
                # 执行推理
                debug_this = (example['id'] == 17)  # 只对ID=17开启调试
                result = is_provable(target, premises, debug=debug_this)
                
                print(f"推理结果: {result}")
                print(f"是否正确: {('True' if result else 'False') == expected_answer}")
                
                if ('True' if result else 'False') == expected_answer:
                    correct_count += 1
            else:
                print("无法解析目标问题")
        else:
            print("无法提取目标问题")
    
    print(f"\n=== 总结 ===")
    print(f"测试总数: {total_count}")
    print(f"正确数量: {correct_count}")
    print(f"准确率: {correct_count / total_count * 100:.1f}%")

# 测试函数
def analyze_failed_example(file_path, example_id):
    """分析特定失败例子的详细信息"""
    data = load_log_data(file_path)
    
    for example in data:
        if example['id'] == example_id:
            print(f"\n=== 分析失败例子 ID={example_id} ===")
            
            question = example['question']
            expected_answer = example['answer']
            
            # 提取条件和目标
            conditions = extract_given_conditions(question)
            target_question = extract_target_question(question)
            
            print(f"预期答案: {expected_answer}")
            print(f"目标问题: {target_question}")
            print(f"条件数量: {len(conditions)}")
            
            print("\n详细条件:")
            for j, condition in enumerate(conditions):
                print(f"  {j+1:2d}. {condition}")
            
            # 解析条件为前提
            premises = []
            for condition in conditions:
                if ' is ' in condition:
                    parts = condition.split(' is ', 1)
                    if len(parts) == 2:
                        input_part = parts[0].strip()
                        output_part = parts[1].strip()
                        premises.append(parse_statement(condition, input_part, output_part))
            
            print(f"\n解析后的前提数量: {len(premises)}")
            
            # 解析目标问题
            if target_question and ' is ' in target_question:
                parts = target_question.split(' is ', 1)
                if len(parts) == 2:
                    input_part = parts[0].strip()
                    output_part = parts[1].strip()
                    target = parse_statement(target_question, input_part, output_part)
                    
                    print(f"\n目标解析:")
                    print(f"  输入: {target['input']}")
                    print(f"  输出: {target['output']}")
                    print(f"  类型: {target['output_parsed']['type']}")
                    print(f"  实体: {target['output_parsed']['entities']}")
                    
                    # 执行推理（开启调试）
                    print(f"\n=== 开始推理 ===")
                    result = is_provable(target, premises, debug=True)
                    
                    print(f"\n推理结果: {result}")
                    print(f"是否正确: {('True' if result else 'False') == expected_answer}")
            
            return
    
    print(f"未找到ID={example_id}的例子")

def test_batch_log_files():
    """批量测试多个LoG数据文件"""
    files_to_test = [
        './generated_data/LoG_4.jsonl',
        './generated_data/LoG_5.jsonl', 
        './generated_data/LoG_6.jsonl',
        './generated_data/LoG_7.jsonl',
        './generated_data/LoG_8.jsonl',
        './generated_data/LoG_9.jsonl',
        './generated_data/LoG_10.jsonl',
        './generated_data/LoG_11.jsonl',
        './generated_data/LoG_12.jsonl'
    ]
    
    total_correct = 0
    total_tested = 0
    results = {}
    
    for file_path in files_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"测试文件: {file_path}")
            print(f"{'='*60}")
            
            # 加载数据
            data = load_log_data(file_path)
            print(f"加载了 {len(data)} 个例子")
            
            # 测试前30个例子
            correct_count = 0
            test_count = min(30, len(data))
            
            for i in range(test_count):
                example = data[i]
                
                question = example['question']
                expected_answer = example['answer']
                
                # 提取条件和目标
                conditions = extract_given_conditions(question)
                target_question = extract_target_question(question)
                
                # 解析条件为前提
                premises = []
                for condition in conditions:
                    if ' is ' in condition:
                        parts = condition.split(' is ', 1)
                        if len(parts) == 2:
                            input_part = parts[0].strip()
                            output_part = parts[1].strip()
                            premises.append(parse_statement(condition, input_part, output_part))
                
                # 解析目标问题
                if target_question and ' is ' in target_question:
                    parts = target_question.split(' is ', 1)
                    if len(parts) == 2:
                        input_part = parts[0].strip()
                        output_part = parts[1].strip()
                        target = parse_statement(target_question, input_part, output_part)
                        
                        # 执行推理（不开启调试以加快速度）
                        result = is_provable(target, premises, debug=False)
                        
                        is_correct = ('True' if result else 'False') == expected_answer
                        if is_correct:
                            correct_count += 1
                        
                        # 只打印错误的例子
                        if not is_correct:
                            print(f"错误例子 {i+1} (ID: {example['id']}): 预期 {expected_answer}, 得到 {result}")
                            print(f"  目标: {target_question}")
                            print(f"  条件数: {len(conditions)}")
                
                # 每10个例子打印一次进度
                if (i + 1) % 10 == 0:
                    print(f"已测试 {i+1}/{test_count} 个例子, 当前正确率: {correct_count/(i+1)*100:.1f}%")
            
            accuracy = correct_count / test_count * 100
            results[file_path] = {
                'correct': correct_count,
                'total': test_count,
                'accuracy': accuracy
            }
            
            total_correct += correct_count
            total_tested += test_count
            
            print(f"\n文件 {file_path} 结果:")
            print(f"  正确数量: {correct_count}/{test_count}")
            print(f"  准确率: {accuracy:.1f}%")
            
        except Exception as e:
            print(f"测试文件 {file_path} 时出错: {e}")
            results[file_path] = {'error': str(e)}
    
    # 打印总结
    print(f"\n{'='*60}")
    print("总结报告")
    print(f"{'='*60}")
    
    for file_path, result in results.items():
        if 'error' in result:
            print(f"{file_path}: 错误 - {result['error']}")
        else:
            print(f"{file_path}: {result['correct']}/{result['total']} ({result['accuracy']:.1f}%)")
    
    if total_tested > 0:
        overall_accuracy = total_correct / total_tested * 100
        print(f"\n总体准确率: {total_correct}/{total_tested} ({overall_accuracy:.1f}%)")
    
    return results

def test_backward_reasoning():
    """测试反向推理系统"""
    
    # 批量测试LoG数据文件
    test_batch_log_files()
    
    print("\n" + "="*50)
    print("以下是原始的单元测试")
    print("="*50)
    
    # 测试1: 简单MP规则
    print("=== 测试1: 简单MP规则 ===")
    premises1 = [
        parse_statement("x is A", "x", "A"),
        parse_statement("A is B", "A", "B")
    ]
    conclusion1 = parse_statement("x is B", "x", "B")
    result1 = is_provable(conclusion1, premises1, debug=True)
    print(f"结果: {result1}\n")
    
    # 测试2: CE规则
    print("=== 测试2: CE规则 ===")
    premises2 = [
        parse_statement("x is A and B", "x", "A and B")
    ]
    conclusion2 = parse_statement("x is A", "x", "A")
    result2 = is_provable(conclusion2, premises2, debug=True)
    print(f"结果: {result2}\n")

def test_single_example():
    """测试单个例子来验证改进效果"""
    # 测试LoG_8的第一个失败例子 (ID=0)
    analyze_failed_example('./generated_data/LoG_8.jsonl', 0)

if __name__ == "__main__":
    print("=== 测试极限深度搜索算法 ===")
    print("激进改进:")
    print("1. 极大增加max_depth: 50→600→2000")
    print("2. 超时时间: 120s（让时间而非深度限制搜索）")
    print("3. **完全移除路径数量限制**")
    print("4. 针对400+条件的超复杂问题:")
    print("   - 重复访问阈值: 50层")
    print("   - 访问记录保留: 40层")
    print("5. 优化路径排序（优先简单路径）")
    print("6. 改进DI规则支持扩展推理")
    print()
    print("策略：给算法最大的探索自由度，")
    print("      只用超时来防止无限运行。")
    print()
    
    # 批量测试所有文件
    test_batch_log_files()