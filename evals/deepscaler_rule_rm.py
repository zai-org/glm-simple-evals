"""
Answer checker API that uses sympy to simplify expressions and check for equality.

Call grade_answer(given_answer: str, ground_truth: str).
"""
from sympy.parsing.sympy_parser import standard_transformations, implicit_multiplication_application, convert_xor
from sympy.parsing.latex import parse_latex
import json
import re
from pylatexenc import latex2text
import sympy
from sympy.parsing import sympy_parser
from typing import Any, List, Optional, Dict
from sympy import sympify, Interval, oo

# from .data_process_utils import (
#     extract_question_and_answer_from_sentence,
# )
# from nemo.utils import logging

# logging.info("DeepscaleR Here!!!")

# Dan Hendrycks' code
def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer

def _strip_string(string):
    def _fix_fracs(string):
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string


    def _fix_a_slash_b(string):
        if len(string.split("/")) != 2:
            return string
        a = string.split("/")[0]
        b = string.split("/")[1]
        try:
            a = int(a)
            b = int(b)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except:
            return string


    def _remove_right_units(string):
        # "\\text{ " only ever occurs (at least in the val set) when describing units
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            assert len(splits) == 2
            return splits[0]
        else:
            return string


    def _fix_sqrt(string):
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    
    # Count the number of \frac commands to handle properly
    frac_count = expr.count("\\frac")
    
    # Use latex2text to convert LaTeX expressions to plain text
    # This handles basic conversion but we need to fix fractions
    original_expr = expr
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    
    # Process each \frac separately by finding numerator and denominator
    # in the original LaTeX expression
    result_expr = expr
    
    # Find all \frac commands in the original expression
    frac_positions = []
    i = 0
    if '\\frac' in original_expr:
        while i < len(original_expr):
            if original_expr[i:i+5] == "\\frac":
                # Found a \frac command
                i += 5
                # Find the numerator (content inside first set of braces)
                if original_expr[i] == '{':
                    i += 1
                    brace_count = 1
                    start = i
                    while i < len(original_expr) and brace_count > 0:
                        if original_expr[i] == '{':
                            brace_count += 1
                        elif original_expr[i] == '}':
                            brace_count -= 1
                        i += 1
                    numerator = original_expr[start:i-1]
                    
                    # Find the denominator (content inside second set of braces)
                    if i < len(original_expr) and original_expr[i] == '{':
                        i += 1
                        brace_count = 1
                        start = i
                        while i < len(original_expr) and brace_count > 0:
                            if original_expr[i] == '{':
                                brace_count += 1
                            elif original_expr[i] == '}':
                                brace_count -= 1
                            i += 1
                        denominator = original_expr[start:i-1]
                        
                        # Store the fraction information
                        frac_positions.append((numerator, denominator))
            else:
                i += 1
    # Now replace each fraction in the converted text with (numerator)/(denominator)
    for numerator, denominator in frac_positions:
        # Convert the LaTeX numerator and denominator to text
        num_text = latex2text.LatexNodes2Text().latex_to_text(numerator)
        denom_text = latex2text.LatexNodes2Text().latex_to_text(denominator)
        
        # Find the fraction in the result text (it's likely in the form num/denom)
        simple_fraction = f"{num_text}/{denom_text}"
        if simple_fraction in result_expr:
            result_expr = result_expr.replace(simple_fraction, f"({num_text})/({denom_text})")
    
    # Replace the specific characters that this parser uses
    result_expr = result_expr.replace("√", "sqrt")
    result_expr = result_expr.replace("π", "pi")
    result_expr = result_expr.replace("∞", "inf")
    result_expr = result_expr.replace("∪", "U")
    result_expr = result_expr.replace("·", "*")
    result_expr = result_expr.replace("×", "*")

    return result_expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _str_is_float(expr: str) -> bool:
    if '.' not in expr:  # 先排除整数
        return False
    return bool(re.match(r'^-?\d*\.\d+$', expr))

def _is_frac(expr: str) -> bool:

    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False

def _float_frac_judge(float_str, frac_str):
    try:
        # 将小数字符串转为float
        float_num = float(float_str)
        
        # 将分数字符串转为float
        # 例如 "3/4" -> num=3, denom=4 -> 3/4=0.75
        num, denom = map(float, frac_str.split('/'))
        frac_num = num / denom
        # 比较两个float是否相等（考虑精度误差）
        return abs(float_num - frac_num) <= 1e-7
    except:
        return False



def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None
    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        try:
            # expr = expr.replace("{", "{\\left \\{")
            # expr = expr.replace("}", "\\right \\}}")
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
                return are_equal
    except:
        pass

    try:
        numeric_tolerance = 0.0000001
        # if float()
        # 将表达式转换为浮点数值
        ground_truth_normalized_n = float(sympy.N(_sympy_parse(ground_truth_normalized)))
        given_normalized_n = float(sympy.N(_sympy_parse(given_normalized)))
        
        expr = f"({ground_truth_normalized_n})-({given_normalized_n})"
        sympy_diff = _sympy_parse(expr)
        simplified = sympy.simplify(sympy_diff)
        if abs(sympy_diff) < numeric_tolerance:
            are_equal = True
            return are_equal
    except:
        pass
    
    # try:
    #     transformations = (standard_transformations + 
    #               (implicit_multiplication_application, convert_xor))
    #     expr = f"({ground_truth_normalized})-({given_normalized})"
    #     expr = expr.replace("=", "==").replace(":=", "==")
    #     sympy_diff = sympy_parser.parse_expr(expr, transformations=transformations)
    #     simplified = sympy.simplify(sympy_diff)
    #     if simplified == 0:
    #         are_equal = True
    #         return are_equal
    # except:
    #     pass

    return are_equal

def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(solution)
    solution = remove_boxed(solution)
    return solution


def _parse_interval(interval_str: str):
    """将区间字符串解析为SymPy的Interval对象"""
    from sympy import Interval, oo, sympify
    
    # 移除所有空格
    interval_str = interval_str.replace(" ", "")
    
    # 确定区间类型（左闭右闭、左闭右开等）
    left_closed = interval_str.startswith('[')
    right_closed = interval_str.endswith(']')
    
    # 提取区间的左右边界
    content = interval_str[1:-1]  # 移除括号
    parts = content.split(',')
    
    if len(parts) != 2:
        raise ValueError(f"Invalid interval format: {interval_str}")
    
    left, right = parts
    
    # 处理无穷大
    if 'inf' in right or '+inf' in right:
        right_val = oo
    else:
        right_val = sympify(right)
    
    if '-inf' in left:
        left_val = -oo
    else:
        left_val = sympify(left)
    
    return Interval(left_val, right_val, left_closed, right_closed)

def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    mark = "no"
    # judge interval first
    try:
        # 尝试将字符串转换为SymPy表达式
        # 对于区间表示，可能需要预处理
        
        # 处理区间表示，例如将[3,+inf)或[3,inf)转换为SymPy的Interval对象
        if '[' in ground_truth or '(' in ground_truth:
            ground_truth_expr = _parse_interval(ground_truth)
        else:
            ground_truth_expr = sympify(ground_truth)
            
        if '[' in given_answer or '(' in given_answer:
            given_expr = _parse_interval(given_answer)
        else:
            given_expr = sympify(given_answer)
        
        # 比较表达式是否相等
        return ground_truth_expr == given_expr, 'interval'
    except Exception as e:
        pass

    if ground_truth_normalized is None:
        return False, mark

    if ground_truth_normalized == given_normalized:
        return True, mark
    if len(given_normalized) == 0:
        return False, mark
    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)
    
    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0]
        or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            # print(ground_truth, given_elem)
            # print("小：", _str_is_float(ground_truth_elem), _str_is_float(given_elem))
            # print("分：",_is_frac(ground_truth_elem), _is_frac(given_elem))
            # print("整：",_str_is_int(ground_truth_elem), _str_is_int(given_elem))
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):   
                mark = "number"
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
                # print("分分:",given_elem, ground_truth_elem, is_correct)
            elif _str_is_int(ground_truth_elem) and _str_is_int(given_elem):
                mark = "number"
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                # print("整整:",given_elem, ground_truth_elem, is_correct)
            # 都是小数的情况
            elif _str_is_float(ground_truth_elem) and _str_is_float(given_elem):
                mark = "number"
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                # print("小小:",given_elem, ground_truth_elem, is_correct)
                
            # 一个是整数，一个是分数的情况
            elif _str_is_int(ground_truth_elem) and _is_frac(given_elem):
                mark = "number"
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                # print("整分:",given_elem, ground_truth_elem, is_correct)
            # 一个是分数，一个是整数的情况
            
            elif _is_frac(ground_truth_elem) and _str_is_int(given_elem):
                mark = "number"
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                # print("分整:",given_elem, ground_truth_elem, is_correct)

            # 一个是整数，一个是小数的情况
            elif _str_is_int(ground_truth_elem) and _str_is_float(given_elem):
                mark = "number"
                is_correct = float(ground_truth_elem) == given_elem
                # print("整小:",given_elem, ground_truth_elem, is_correct)
            # 一个是小数，一个是整数的情况
            elif _str_is_float(ground_truth_elem) and _str_is_int(given_elem):
                mark = "number"
                is_correct = float(given_elem) == ground_truth_elem
                # print("小整:",given_elem, ground_truth_elem, is_correct)

            # 一个是小数，一个是分数的情况
            elif _is_float(ground_truth_elem) and _is_frac(given_elem):
                mark = "number"
                is_correct = _float_frac_judge(ground_truth_elem, given_elem)
                # print("小分:",given_elem, ground_truth_elem, is_correct)
            # 一个是分数，一个是小数的情况
            elif _is_frac(ground_truth_elem) and _is_float(given_elem):
                mark = "number"
                is_correct = _float_frac_judge(given_elem, ground_truth_elem)
                # print("分小:",given_elem, ground_truth_elem, is_correct)


            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break
    return is_correct, mark


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    ground_truth_normalized_mathd = mathd_normalize_answer(ground_truth)
    given_answer_normalized_mathd = mathd_normalize_answer(given_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True
    return False

def extract_answer(passage: str) -> str:
    if "\\boxed" in passage:
        return extract_boxed_answer(passage)
    return None

def grade_answer_verl(solution_str, ground_truth):
    if not ground_truth:
        return False
    if '\\boxed' in ground_truth:
        ground_truth = extract_answer(ground_truth)
    given_answer = extract_answer(solution_str)
    if given_answer is None:
        return False
    return grade_answer_mathd(given_answer, ground_truth) \
        or grade_answer_sympy(given_answer, ground_truth)

def _get_deepscaler_rule_base_reward(question, response, label):
    model_answer = response

    if model_answer is None:
        return 0
    if label == "":
        return 0
    if isinstance(label, (str, float, int)):
        ground_truths = [label]
    else:
        return 0
    for ground_truth in ground_truths:
        is_correct_1 = grade_answer_mathd(model_answer, ground_truth)
        is_correct_2, mark = grade_answer_sympy(model_answer, ground_truth)
        is_correct =  is_correct_1 or is_correct_2
        if is_correct:
            return 1
        elif not is_correct and mark=="number":
            return 0
        
    return -1
        

def _get_deepscaler_rule_base_reward_with_extract(question, response, label):
    model_solution = response
    model_answer = extract_answer(model_solution)
    if model_answer is None:
        return 0
    if label == "":
        return 0
    # Convert single answer to list for uniform processing
    if isinstance(label, (str, float, int)):
        ground_truths = [label]
        
    # Process each ground truth
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    
    if not processed_ground_truths:
        return 0
    
    # Check against all possible correct answers
    for ground_truth in processed_ground_truths:
        is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
        if is_correct:
            return 1
            
    return 0

def is_int_colon_int(s):
    pattern = r"^\s*\d+\s*:\s*\d+\s*$"
    return bool(re.match(pattern, s))
    
def convert_colon_to_frac(text):
    def replace_match(match):
        # 提取整数部分并去除空格
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        return f"\\frac{{{numerator}}}{{{denominator}}}"
    
    # 匹配数字、冒号、数字的模式，允许周围有空格
    pattern = r"(\d+)\s*:\s*(\d+)"
    
    # 使用正则表达式替换所有匹配项
    result = re.sub(pattern, replace_match, text)
    return result

def send_deepscaler_rule_rm_request(answer,label,response=None) :

    if is_int_colon_int(answer):
        try: 
            # 分割并移除空格
            parts = answer.split(":")
            num = int(parts[0].strip())
            denom = int(parts[1].strip())          
            simplify = num % denom
            
            # 如果结果是整数
            if simplify == 0:
                answer = str(int(num//denom))
            else:
                answer = f"{num}/{denom}"
        except Exception as e:
            pass
        
    if is_int_colon_int(label):
        try: 
            # 分割并移除空格
            parts = label.split(":")
            num = int(parts[0].strip())
            denom = int(parts[1].strip())        
            simplify = num % denom
            
            # 如果结果是整数
            if simplify == 0:
                label = str(int(num//denom))
            else:
                label = f"{num}/{denom}"
        except Exception as e:
            pass
    rule_reward = _get_deepscaler_rule_base_reward("testtesttest", answer, label)
    # print(answer,"|||", label, "|||", rule_reward)
    return rule_reward