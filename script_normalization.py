import re
from random import randint

SOLUGATE_SPECIAL_SYMBOLS = re.compile(r"[/*+blon.?]")
DIQUEST_SPECIAL_SYMBOLS = re.compile(r"[*FfNnOoPpSs:]")
SPACES = re.compile(r"\s+")
SLASH_SEPARATED_PARENS = re.compile("(\([^()]*\)/\([^()]*\))")
INSIDE_PARANS = re.compile("\(([^()]*)\)")


def _is_parentheses_matching_error(text: str):
    left = text.count('(')
    right = text.count(')')
    
    if(left == right):
        return False
    
    return True


def _spelling_rep(text: str):
    result = ""
    
    segment_list = SLASH_SEPARATED_PARENS.split(text)
    if(len(segment_list) != 1):
        for segment in segment_list:
            curr = INSIDE_PARANS.findall(segment)
            if(len(curr) == 0):
                result += segment
            else:
                if(bool(re.search('\d', curr[0])) and bool(randint(0,1))):
                    try:
                        result += curr[1]
                    except:
                        print(text)
                else:
                    result += curr[0]
        return result
    
    return text


def _is_parentheses_parse_error(text: str):
    is_error = False
    partitions = SLASH_SEPARATED_PARENS.split(text)
    if(len(partitions) != 1):
        for part in partitions:
            curr = INSIDE_PARANS.findall(part)
            for element in curr:
                if(('(' in element) or (')' in element)):
                    is_error = True
                    break
                
    return is_error


def solugate_speech_normalize(text: str):
    '''
        1. Parantheses inside parantheses check
        2. Check for incorrect paranthese count match
        3. 
    '''

    error = _is_parentheses_parse_error(text)
    if error:
        return None

    text = _spelling_rep(text)
    
    # if(bool(re.search('\d', line_of_text))):
    #     return None
    
    error = _is_parentheses_matching_error(text)
    if error:
        return None

    text = SOLUGATE_SPECIAL_SYMBOLS.sub("", text)
    text = SPACES.sub(" ", text).strip()
    
    if(len(text) < 5 or len(text) > 700):
        return None

    return text


def diquest_speech_normalize(text: str):
    if INSIDE_PARANS.search(text) is not None:
        partition_list = INSIDE_PARANS.split(text)
        for partition in partition_list:
            if "FP" not in partition:
                partition = partition.split(":",1)[-1]
            
        text = "".join(partition_list)
        
    text = DIQUEST_SPECIAL_SYMBOLS.sub("",text)
    text = SPACES.sub(" ",text).strip()
    
    return text