import argparse
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random

# @dataclass
# class LogicNode:
#     """逻辑节点类，表示一个逻辑语句"""
    
#     id: str  # 节点唯一标识符
#     description: str  # 自然语言描述，如 "A and B are both C"
#     node_type: str = "logic"  # 节点类型，可以是 "logic", "input", "output" 等
    
#     # 可选的元数据
#     metadata: Dict[str, Any] = field(default_factory=dict)
    
#     def __post_init__(self):
#         """初始化后的验证"""
#         if not self.id:
#             raise ValueError("Node ID cannot be empty")
#         if not self.description:
#             raise ValueError("Node description cannot be empty")
    
#     def __str__(self):
#         return f"LogicNode(id={self.id}, description='{self.description}')"
    
#     def __repr__(self):
#         return self.__str__()
    
# class Hop:
#     output: str
#     input: list
#     deduction_rule: str
#     depth: int
def copy_ls(ls_orig):
    ls_new = []
    for i in ls_orig:
        ls_new.append(i)
    return ls_new

element_ls = ['brimpus', 'gorpus', 'shumpus', 'vumpus', 'numpus', 'dumpus', 'sterpus', 'jorpus', 'yumpus', 'impus', 'rompus', ' wumpus', 'jompus', 'grimpus', 'frompus', 'himpus', 'anpus', 'korpus', 'lorpus', 'mumpus', 'orpus', 'porpus', 'quampus', 'sampus', 'tinpus', 'unpus', 'zinpus'] ##主语列表
deduction_rule_ls = ['CE', 'DI', 'CI', 'MP'] ##演绎规则列表
conjunction = ['and', 'or'] ##连接符列表 与或非，非随机加在主语前或句子前
refuse_ls = [] ##要考虑主语之间的重合程度，不可能不同hop间完全不用相同的主语，但是重复也有可能造成逻辑环

# element_ind_dict = {}## 主语和序号的映射关系，brimpus:1 gorpus:2，用于pop重复的主语
# for i in range(0, len(element_ls)):
#     element_ind_dict[element_ls[i]] = i
def MP_generate(tmp_output):#MP的输出是x is A, 输入是x is B和 B is A，返回
    return [1, 2]
def input_generate(pre_conjunction, len_output, tmp_output):
    tmp_deduction_rule = ['MP']#什么时候都能有MP
    if(pre_conjunction == "or"):
        tmp_deduction_rule.append('DI')
    if(pre_conjunction == 'and'):
        tmp_deduction_rule.append('CI')
    if((pre_conjunction == 'and' and len_output < 4) or len_output == 1):
        tmp_deduction_rule.append('CE')
    chosen_deduction_rule = tmp_deduction_rule[random.randint(0, len(tmp_deduction_rule)-1)]
    #####接下来补充各个deduction_rule的逻辑代码
    tmp_res = None
    if(chosen_deduction_rule == 'MP'):
        tmp_res = MP_generate(tmp_output)
        return ['MP', tmp_res[0], tmp_res[1]]
    ##下面别的dedution_rule follow上面的，就是根据需要的信息生成可能的输入，返回


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hop', type=int, default=2)

    args = parser.parse_args()

    # print(args.num_hop)
    ls_hop_dealing = []##ls_hop是一个先进先出的队列，用于维护需要继续生长的叶子结点,该列表为空时，判断树生长结束
    ls_hop_res = []##ls_hop_res用于存储每一个hop
    flag_start = True
    element_dynamic = copy_ls(element_ls)##这个程序生成单点数据，这个函数没有用；但是生成多个数据时能保证源element_ls稳定，所以先写在这里
    while True:
        if(flag_start == True):#第一轮只是生成 待证明的结论，但是由于形式上需要统一成同一个字典形式，所以有这个处理逻辑
            tmp_dict = {}
            tmp_element_cnt = random.randint(1,3)
            tmp_conjunction = conjunction[random.randint(0,1)]
            if(tmp_element_cnt == 1):
                tmp_conjunction = ''
            tmp_dict['input'] = None
            tmp_output = 'x is ' + element_dynamic[0]
            for i in range(1, tmp_element_cnt):
                tmp_output += ' ' + tmp_conjunction + ' ' + element_dynamic[i]
            tmp_dict['output'] = tmp_output
            tmp_dict['deduction_rule'] = None
            tmp_dict['conjunction'] = tmp_conjunction
            tmp_dict['depth'] = 1
            flag_start = False
            ls_hop_dealing.append(tmp_dict)
        else:
            if(len(ls_hop_dealing) == 0):#说明遍历结束，LoG图生成完成
                break
            else:#还有待生长节点
                pre_dict = ls_hop_dealing[0]
                if(pre_dict['depth'] > args.num_hop):##当前节点已经达到了用户要求的深度，不做处理
                    pass
                else:##图还没有达到用户要求的深度，继续生长
                    tmp_dict = {}
                    ls_hop_dealing.pop(0)
                    pre_conjunction = pre_dict['conjunction']
                    pre_depth = pre_dict['depth']
                    tmp_output = pre_dict['output']
                    tmp_dict['output'] = tmp_output
                    len_output = len(tmp_output.split(pre_conjunction))
                    tmp_res = input_generate(pre_conjunction, len_output, tmp_output)
                    tmp_dict['conjunction'] = pre_conjunction
                    tmp_dict['deduction_rule'] = tmp_res[0]
                    tmp_dict['depth'] = pre_depth + 1
                    tmp_dict['input'] = tmp_res[1:]##除了第一个是deduction rule，别的都是生成的输入
                    ls_hop_res.append(tmp_dict)
                    for i in tmp_dict['input']:
                        tmp_todo_dict = {}
                        tmp_todo_dict['output'] = i
                        tmp_todo_dict['depth'] = pre_depth + 1
                        tmp_todo_dict['deduction_rule'] = None
                        tmp_todo_dict['input'] = None
                        if('and' in tmp_todo_dict['output']):
                            tmp_todo_dict['conjunction'] = "and"
                        elif('or' in tmp_todo_dict['output']):
                            tmp_todo_dict['conjunction'] = 'or'
                        else:
                            tmp_todo_dict['conjunction'] = ''


            
    print(ls_hop_res)






    