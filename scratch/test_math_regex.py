import re

# Robust regex pattern without restricting word boundaries on special characters like %
math_pattern = r'(\d+[\d\.,]*)\s*(?:đồng|đ|USD|%)?\s*([\*xX\/+-])\s*(\d+[\d\.,]*%?)\s*=\s*(\d+[\d\.,]*)'

test_cases = [
    "500.000 đồng * 300% = 1.500.000 đồng",
    "500.000đ * 300% = 1.500.000đ",
    "500.000 * 3 = 1.500.000",
    "500,000đ x 300% = 1,500,000đ",
    "500.000 đồng * 3 = 1.500.000đ",
    "500.000 * 300% = 1.500.000đ",
    "Lương 10.5 triệu * 1.5 = 15.75 triệu", # testing decimal
]

def clean_num(s):
    s = s.strip()
    if s.endswith('%'):
        s = s[:-1]
    
    # If both dot and comma are present
    if '.' in s and ',' in s:
        # Determine which one is the decimal point (the last one)
        if s.rfind('.') > s.rfind(','):
            # Comma is thousands, dot is decimal
            s = s.replace(',', '')
        else:
            # Dot is thousands, comma is decimal
            s = s.replace('.', '').replace(',', '.')
        return float(s)
        
    # If only one type of separator is present
    separator = None
    if '.' in s:
        separator = '.'
    elif ',' in s:
        separator = ','
        
    if separator:
        parts = s.split(separator)
        # If there are multiple separators, it's definitely thousands
        if len(parts) > 2:
            s = s.replace(separator, '')
        else:
            # Exactly one separator. Check digits after it.
            # If exactly 3 digits, it's thousands (e.g. 500.000)
            if len(parts[1]) == 3:
                s = s.replace(separator, '')
            else:
                # E.g. 1.5 or 15,75
                s = s.replace(separator, '.')
                
    return float(s)

output_lines = []
output_lines.append("Regex Matching Results:")
for case in test_cases:
    matches = re.findall(math_pattern, case)
    output_lines.append(f"Case: {case}")
    output_lines.append(f"Matches: {matches}")
    if matches:
        op1_str, operator, op2_str, res_str = matches[0]
        try:
            val1 = clean_num(op1_str)
            val2 = clean_num(op2_str)
            val_res = clean_num(res_str)
            
            is_pct = '%' in op2_str
            multiplier = val2 / 100.0 if is_pct else val2
            
            calculated_res = 0.0
            op = operator.lower()
            if op in ('*', 'x'):
                calculated_res = val1 * multiplier
            elif op == '/':
                calculated_res = val1 / multiplier if multiplier != 0 else 0.0
            elif op == '+':
                calculated_res = val1 + multiplier
            elif op == '-':
                calculated_res = val1 - multiplier
                
            diff = abs(calculated_res - val_res)
            is_valid = diff <= 1.0
            output_lines.append(f"  Parsed: op1={val1}, op2={val2} (pct={is_pct}), res={val_res}")
            output_lines.append(f"  Calculated: {calculated_res}, Diff: {diff} -> {'VALID' if is_valid else 'INVALID'}")
        except Exception as e:
            output_lines.append(f"  Error parsing/calculating: {e}")
    output_lines.append("")

with open("scratch/math_regex_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print("Success! Output written to scratch/math_regex_output.txt")
