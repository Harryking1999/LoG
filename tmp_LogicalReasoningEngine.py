class LogicalReasoningEngine:
    """逻辑推理引擎 - 集成自debug_test.py的推理算法"""
    
    def __init__(self, max_depth: int = 2000, timeout: int = 120):
        """
        初始化推理引擎
        
        Args:
            max_depth: 最大推理深度
            timeout: 超时时间（秒）
        """
        self.max_depth = max_depth
        self.timeout = timeout
    
    def parse_output_entities(self, output_part: str) -> Dict[str, Any]:
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
    
    def statements_equal(self, stmt1: Dict[str, Any], stmt2: Dict[str, Any]) -> bool:
        """判断两个语句是否相等"""
        return (stmt1["input"] == stmt2["input"] and 
                stmt1["output"] == stmt2["output"])
    
    def is_provable(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                   visited: set = None, depth: int = 0, start_time: float = None, 
                   debug: bool = False) -> bool:
        """
        反向推理：判断目标是否可以从前提中推导出来
        
        Args:
            target: 目标结论，格式为节点字典
            premises: 前提条件列表，每个元素为节点字典
            visited: 已访问的目标集合，防止循环
            depth: 递归深度
            start_time: 开始时间
            debug: 是否打印调试信息
            
        Returns:
            是否可以推导出来
        """
        if visited is None:
            visited = set()
        
        if start_time is None:
            start_time = time.time()
        
        # 超时检查
        if time.time() - start_time > self.timeout:
            if debug:
                print(f"推理超时({self.timeout}s)，终止")
            return False
        
        indent = "  " * depth
        if debug:
            print(f"{indent}尝试证明: {target['input']} is {target['output']}")
        
        # 改进的循环检测：防止无限递归
        target_key = f"{target['input']}→{target['output']}"
        
        # 如果在当前路径中已经访问过这个目标，直接跳过
        if target_key in visited:
            if debug:
                print(f"{indent}检测到循环依赖，跳过")
            return False
        
        if depth > self.max_depth:
            if debug:
                print(f"{indent}超过最大深度({self.max_depth})，跳过")
            return False
        
        # 临时添加到visited
        visited.add(target_key)
        
        try:
            # 基础情况：目标已经在前提中
            for premise in premises:
                if self.statements_equal(target, premise):
                    if debug:
                        print(f"{indent}✓ 在前提中找到: {premise.get('original', premise['input'] + ' is ' + premise['output'])}")
                    return True
            
            # 寻找可能的推理路径
            possible_paths = self.find_reasoning_paths(target, premises, debug and depth < 5)
            
            if debug:
                print(f"{indent}找到 {len(possible_paths)} 种推理路径")
            
            # 按优先级排序路径：优先尝试简单的路径
            def path_priority(path):
                rule_priority = {
                    'CE': 1, 'DI_EXPAND': 2, 'DI': 3, 'MP': 4, 'CI': 5, 'MP+CE': 6
                }
                return (len(path['intermediates']), rule_priority.get(path['rule'], 10))
            
            possible_paths.sort(key=path_priority)
            
            # 尝试所有推理路径，只受超时限制
            for i, path in enumerate(possible_paths):
                # 超时检查
                if time.time() - start_time > self.timeout:
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
                    if not self.is_provable(intermediate, premises, visited.copy(), 
                                          depth + 1, start_time, debug):
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
    
    def find_reasoning_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                           debug: bool = False) -> List[Dict[str, Any]]:
        """从已有前提中寻找可以推导出目标的推理路径"""
        paths = []
        target_input = target["input"]
        target_output = target["output"]
        target_output_parsed = target["output_parsed"]
        
        # 规则1: MP (Modus Ponens)
        mp_paths = self.find_mp_paths(target, premises, debug)
        paths.extend(mp_paths)
        
        # 规则2: CE (Conjunction Elimination) 
        ce_paths = self.find_ce_paths(target, premises, debug)
        paths.extend(ce_paths)
        
        # 规则3: CI (Conjunction Introduction) - 只对and类型的目标应用
        if target_output_parsed["type"] == "and":
            ci_paths = self.find_ci_paths(target, premises, debug)
            paths.extend(ci_paths)
        
        # 规则4: DI (Disjunction Introduction) - 只对or类型的目标应用
        if target_output_parsed["type"] == "or":
            di_paths = self.find_di_paths(target, premises, debug)
            paths.extend(di_paths)
        
        return paths
    
    def find_mp_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
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
                        "output_parsed": target_output_parsed,
                        "original": f"{x_value} is {target_output}",
                        "type": "intermediate"
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
                            "output_parsed": {"type": "single", "entities": [entity]},
                            "original": f"{target_input} is {entity}",
                            "type": "intermediate"
                        }
                        mp_intermediate = {
                            "input": entity,
                            "output": target_output,
                            "output_parsed": target_output_parsed,
                            "original": f"{entity} is {target_output}",
                            "type": "intermediate"
                        }
                        
                        paths.append({
                            "rule": "MP+CE",
                            "intermediates": [ce_intermediate, mp_intermediate]
                        })
                        
                        if debug:
                            print(f"    MP+CE路径: {target_input} is {x_value} → CE得到 {target_input} is {entity} → MP需要 {entity} is {target_output}")
        
        return paths
    
    def find_ce_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
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
                        "output_parsed": premise["output_parsed"],
                        "original": premise.get("original", f"{target_input} is {premise['output']}"),
                        "type": "intermediate"
                    }
                    paths.append({
                        "rule": "CE",
                        "intermediates": [intermediate]
                    })
                    
                    if debug:
                        print(f"    CE路径: {premise['output']} contains {target_output}")
        
        return paths
    
    def find_ci_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
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
                    "output_parsed": {"type": "single", "entities": [entity]},
                    "original": f"{target_input} is {entity}",
                    "type": "intermediate"
                }
                intermediates.append(intermediate)
            
            paths.append({
                "rule": "CI",
                "intermediates": intermediates
            })
            
            if debug:
                print(f"    CI路径: 需要 {len(entities)} 个单独的条件")
        
        return paths
    
    def find_di_paths(self, target: Dict[str, Any], premises: List[Dict[str, Any]], 
                     debug: bool = False) -> List[Dict[str, Any]]:
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
                    "output_parsed": {"type": "single", "entities": [entity]},
                    "original": f"{target_input} is {entity}",
                    "type": "intermediate"
                }
                
                paths.append({
                    "rule": "DI",
                    "intermediates": [intermediate]
                })
                
                if debug:
                    print(f"    DI路径: 只需要证明 {target_input} is {entity}")
            
            # 方法2: 寻找包含目标实体子集的or条件（DI扩展）
            for premise in premises:
                if (premise["input"] == target_input and 
                    premise["output_parsed"]["type"] == "or"):
                    premise_entities = set(premise["output_parsed"]["entities"])
                    
                    # 如果前提的实体是目标实体的子集，可以通过DI扩展
                    if premise_entities.issubset(target_entities) and premise_entities != target_entities:
                        intermediate = {
                            "input": target_input,
                            "output": premise["output"],
                            "output_parsed": premise["output_parsed"],
                            "original": premise.get("original", f"{target_input} is {premise['output']}"),
                            "type": "intermediate"
                        }
                        
                        paths.append({
                            "rule": "DI_EXPAND",
                            "intermediates": [intermediate]
                        })
                        
                        if debug:
                            print(f"    DI扩展路径: 从 {target_input} is {premise['output']} 扩展到 {target['output']}")
        
        return paths