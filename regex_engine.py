# regex_engine.py

def add_concat_ops(p):
    res = []
    for i in range(len(p)):
        res.append(p[i])
        if i + 1 < len(p):
            c1, c2 = p[i], p[i+1]
            if (c1.isalnum() or c1 in '*)') and (c2.isalnum() or c2 == '('):
                res.append('.')
    return "".join(res)

def to_postfix(p):
    output, stack = [], []
    precedence = {'*': 3, '.': 2, '|': 1}
    for char in p:
        if char.isalnum(): output.append(char)
        elif char == '(': stack.append(char)
        elif char == ')':
            while stack and stack[-1] != '(': output.append(stack.pop())
            stack.pop()
        else:
            while stack and stack[-1] != '(' and precedence.get(stack[-1], 0) >= precedence.get(char, 0):
                output.append(stack.pop())
            stack.append(char)
    while stack: output.append(stack.pop())
    return output

def build_thompson_nfa(pattern):
    state_idx = 0
    def get_state():
        nonlocal state_idx
        s = f"q{state_idx}"
        state_idx += 1
        return s

    stack = []
    postfix = to_postfix(add_concat_ops(pattern))
    
    for char in postfix:
        if char.isalnum():
            s1, s2 = get_state(), get_state()
            stack.append({'start': s1, 'accept': s2, 'edges': [{'source': s1, 'target': s2, 'label': char}]})
        elif char == '.':
            r2, r1 = stack.pop(), stack.pop()
            r1['edges'].extend(r2['edges'])
            r1['edges'].append({'source': r1['accept'], 'target': r2['start'], 'label': 'ε'})
            stack.append({'start': r1['start'], 'accept': r2['accept'], 'edges': r1['edges']})
        elif char == '|':
            r2, r1 = stack.pop(), stack.pop()
            s, a = get_state(), get_state()
            edges = r1['edges'] + r2['edges']
            edges.extend([{'source': s, 'target': r1['start'], 'label': 'ε'},
                          {'source': s, 'target': r2['start'], 'label': 'ε'},
                          {'source': r1['accept'], 'target': a, 'label': 'ε'},
                          {'source': r2['accept'], 'target': a, 'label': 'ε'}])
            stack.append({'start': s, 'accept': a, 'edges': edges})
        elif char == '*':
            r = stack.pop()
            s, a = get_state(), get_state()
            edges = r['edges']
            edges.extend([{'source': s, 'target': r['start'], 'label': 'ε'},
                          {'source': s, 'target': a, 'label': 'ε'},
                          {'source': r['accept'], 'target': r['start'], 'label': 'ε'},
                          {'source': r['accept'], 'target': a, 'label': 'ε'}])
            stack.append({'start': s, 'accept': a, 'edges': edges})

    return stack.pop()