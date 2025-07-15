import random
print(random.randint(0, 2))
# ls = [1,2,3,4,5]
# ls.pop(0)
# print(ls)

# element_ls = ['brimpus', 'gorpus', 'shumpus', 'vumpus', 'numpus', 'dumpus', 'sterpus', 'jorpus', 'yumpus', 'impus', 'rompus', ' wumpus', 'jompus', 'grimpus', 'frompus', 'himpus', 'anpus', 'korpus', 'lorpus', 'mumpus', 'orpus', 'porpus', 'quampus', 'sampus', 'tinpus', 'unpus', 'zinpus'] ##主语列表
# deduction_rule_ls = ['CE', 'DI', 'CI', 'MP'] ##演绎规则列表
# conjunction = ['and', 'or'] ##连接符列表 与或非，非随机加在主语前或句子前
# refuse_ls = [] ##要考虑主语之间的重合程度，不可能不同hop间完全不用相同的主语，但是重复也有可能造成逻辑环

# element_ind_dict = {}## 主语和序号的映射关系，brimpus:1 gorpus:2，用于pop重复的主语
# for i in range(0, len(element_ls)):
#     element_ind_dict[element_ls[i]] = i

# print(element_ind_dict)
