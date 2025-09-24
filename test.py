def is_provable(conclusion, premises):
    """
    递归函数，判断结论是否可以从前提中证明得到
    
    Args:
        conclusion: 目标结论的字典格式
        premises: 前提条件的列表，每个元素都是字典格式
    
    Returns:
        bool: True表示可以证明，False表示无法证明
    """
    # 基础情况：如果结论已经在前提中，直接返回True
    for premise in premises:
        if statements_equal(conclusion, premise):
            return True
    
    # 找到所有可能的中间结论
    possible_conclusions = find_possible_conclusions(conclusion, premises)
    
    # 对每个可能的中间结论进行递归证明
    for intermediate_conclusions in possible_conclusions:
        # 检查是否所有中间结论都可以被证明
        all_provable = True
        for intermediate in intermediate_conclusions:
            if not is_provable(intermediate, premises):
                all_provable = False
                break
        
        if all_provable:
            return True
    
    return False


def find_possible_conclusions(target, premises):
    """
    根据4种推理规则，找到可能推导出目标结论的中间结论组合
    
    Returns:
        List[List]: 每个元素是一个中间结论的列表，如果这些中间结论都能证明，就能证明目标
    """
    possible = []
    
    target_input = target["input"]
    target_output = target["output_parsed"]
    
    # 规则1: MP (Modus Ponens)
    # A is B, B is C -> A is C
    # A is B, B is C and D -> A is C and D
    possible.extend(find_mp_conclusions(target, premises))
    
    # 规则2: CE (Conjunction Elimination)
    # A is B and C -> A is B (反向：要证明A is B，需要A is B and C)
    possible.extend(find_ce_conclusions(target, premises))
    
    # 规则3: CI (Conjunction Introduction) 
    # x is A, x is B -> x is A and B (反向：要证明x is A and B，需要x is A和x is B)
    possible.extend(find_ci_conclusions(target, premises))
    
    # 规则4: DI (Disjunction Introduction)
    # x is A -> x is A or B (反向：要证明x is A or B，需要x is A或x is B中的一个)
    possible.extend(find_di_conclusions(target, premises))
    
    return possible


def find_mp_conclusions(target, premises):
    """找到MP规则的可能中间结论"""
    possible = []
    target_input = target["input"]
    target_output = target["output_parsed"]
    
    # 寻找形如 target_input is X 的前提，然后需要 X is target_output
    for premise in premises:
        if premise["input"] == target_input:
            # 找到了 target_input is X，现在需要 X is target_output
            x_output = premise["output_parsed"]
            
            # 构造需要的中间结论 X is target_output
            needed_conclusion = {
                "input": premise["output"],  # X的字符串形式
                "output_parsed": target_output,
                "output": format_output(target_output)
            }
            
            possible.append([needed_conclusion])
    
    return possible


def find_ce_conclusions(target, premises):
    """找到CE规则的可能中间结论（反向推理）"""
    possible = []
    target_input = target["input"] 
    target_output = target["output_parsed"]
    
    # 要证明 A is B，可能需要 A is B and C 或 A is C and B
    if target_output["type"] != "and":
        # 构造包含目标的and语句
        # 可能的形式：A is target_output and X
        for premise in premises:
            if premise["input"] == target_input and premise["output_parsed"]["type"] == "and":
                entities = premise["output_parsed"]["entities"]
                if target_output["type"] == "single":
                    target_entity = target_output["entities"][0] if "entities" in target_output else target["output"]
                    if target_entity in entities:
                        possible.append([premise])
                elif target_output["type"] == "or":
                    # 检查or中的实体是否都在and中
                    if all(entity in entities for entity in target_output["entities"]):
                        possible.append([premise])
    
    return possible


def find_ci_conclusions(target, premises):
    """找到CI规则的可能中间结论（反向推理）"""
    possible = []
    target_input = target["input"]
    target_output = target["output_parsed"]
    
    # 要证明 x is A and B，需要 x is A 和 x is B
    if target_output["type"] == "and":
        needed_conclusions = []
        for entity in target_output["entities"]:
            conclusion = {
                "input": target_input,
                "output": entity,
                "output_parsed": {"type": "single", "entities": [entity]}
            }
            needed_conclusions.append(conclusion)
        
        if len(needed_conclusions) > 1:
            possible.append(needed_conclusions)
    
    return possible


def find_di_conclusions(target, premises):
    """找到DI规则的可能中间结论（反向推理）"""
    possible = []
    target_input = target["input"]
    target_output = target["output_parsed"]
    
    # 要证明 x is A or B，只需要证明 x is A 或 x is B 中的一个
    if target_output["type"] == "or":
        for entity in target_output["entities"]:
            conclusion = {
                "input": target_input,
                "output": entity,
                "output_parsed": {"type": "single", "entities": [entity]}
            }
            possible.append([conclusion])
    
    return possible


def statements_equal(stmt1, stmt2):
    """判断两个语句是否相等"""
    return (stmt1["input"] == stmt2["input"] and 
            stmt1["output"] == stmt2["output"])


def format_output(output_parsed):
    """将解析后的输出转换为字符串格式"""
    if output_parsed["type"] == "single":
        return output_parsed["entities"][0]
    elif output_parsed["type"] == "and":
        return " and ".join(output_parsed["entities"])
    elif output_parsed["type"] == "or":
        return " or ".join(output_parsed["entities"])
    return ""


def parse_statement(statement_str, input_part, output_part):
    """解析语句字符串，构造标准格式的字典"""
    output_parsed = {"type": "single", "entities": [output_part]}
    
    if " and " in output_part:
        entities = [e.strip() for e in output_part.split(" and ")]
        output_parsed = {"type": "and", "entities": entities}
    elif " or " in output_part:
        entities = [e.strip() for e in output_part.split(" or ")]
        output_parsed = {"type": "or", "entities": entities}
    else:
        output_parsed = {"type": "single", "entities": [output_part]}
    
    return {
        "original": statement_str,
        "input": input_part,
        "output": output_part,
        "output_parsed": output_parsed,
        "type": "actual"
    }


# 测试示例
def test_proof_system():
    """测试逻辑证明系统"""
    
    # 示例1: 简单的MP规则测试
    premises = [
        parse_statement("x is A", "x", "A"),
        parse_statement("A is B", "A", "B")
    ]
    conclusion = parse_statement("x is B", "x", "B")
    
    result1 = is_provable(conclusion, premises)
    print(f"测试1 - MP规则: {result1}")  # 应该返回True
    
    # 示例2: CI规则测试  
    premises2 = [
        parse_statement("x is A", "x", "A"),
        parse_statement("x is B", "x", "B")
    ]
    conclusion2 = parse_statement("x is A and B", "x", "A and B")
    
    result2 = is_provable(conclusion2, premises2)
    print(f"测试2 - CI规则: {result2}")  # 应该返回True
    
    # 示例3: 复合推理测试
    premises3 = [
        parse_statement("x is A", "x", "A"),
        parse_statement("A is B and C", "A", "B and C")
    ]
    conclusion3 = parse_statement("x is B", "x", "B")
    
    result3 = is_provable(conclusion3, premises3)
    print(f"测试3 - MP+CE规则: {result3}")  # 应该返回True
    
    # 示例4: 无法证明的情况
    premises4 = [
        parse_statement("x is A", "x", "A"),
        parse_statement("y is B", "y", "B")
    ]
    conclusion4 = parse_statement("x is B", "x", "B")
    
    result4 = is_provable(conclusion4, premises4)
    print(f"测试4 - 无法证明: {result4}")  # 应该返回False

    premises5 = [
        parse_statement("A is B", "A", "B"),
        parse_statement("B is E", "B", "E"),
        parse_statement("x is A", "x", "A"),
        parse_statement("E is C", "E", "C")
    ]
    conclusion5 = parse_statement("x is D", "x", "D")
    result5 = is_provable(conclusion5, premises5)
    print(f"测试5 - : {result5}")  # 应该返回False

    premises6 = [
        parse_statement("tengpus is ridspus and kixlpus", "tengpus", "ridspus and kixlpus"),
        parse_statement("x is fobvpus", "x", "fobvpus"),
        parse_statement("zipmpus is cawypus and nafgpus", "zipmpus", "cawypus and nafgpus"),
        parse_statement("ridspus is vigdpus", "ridspus", "vigdpus"),
        parse_statement("kagkpus is roqvpus", "kagkpus", "roqvpus"),
        parse_statement("qactpus is tengpus", "qactpus", "tengpus"),
        parse_statement("cawypus is babcpus and galqpus and xilppus", "cawypus", "babcpus and galqpus and xilppus"),
        parse_statement("lesfpus is goxfpus", "lesfpus", "goxfpus"),
        parse_statement("goxfpus is malcpus and yinzpus", "goxfpus", "malcpus and yinzpus"),
        parse_statement("vigdpus is qiflpus", "vigdpus", "qiflpus"),
        parse_statement("roqvpus is lesfpus", "roqvpus", "lesfpus"),
        parse_statement("malcpus is cotwpus", "malcpus", "cotwpus"),
        parse_statement("qiflpus is kagkpus", "qiflpus", "kagkpus"),
        parse_statement("cotwpus is zipmpus and kuvspus", "cotwpus", "zipmpus and kuvspus"),
        parse_statement("fobvpus is qactpus and lirkpus", "fobvpus", "qactpus and lirkpus")
    ]
    conclusion6 = parse_statement("x is babcpus", "x", "babcpus")
    result6 = is_provable(conclusion6, premises6)
    print(f"测试6 - 复杂推理链: {result6}") #应该返回true

if __name__ == "__main__":
    test_proof_system()