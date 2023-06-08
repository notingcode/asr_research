import re
from random import randint

ETRI_SPECIAL_SYMBOLS = re.compile(r"[{}/*+.,?]|b/|l/|o/|n/|u/")
DIQUEST_SPECIAL_SYMBOLS = re.compile(r"[*FfNnOoPpSs:.,?]")
SPACES = re.compile(r"\s+")
SLASH_SEPARATED_PARENS = re.compile("(\([^()]*\)/\([^()]*\))")
INSIDE_PARANS = re.compile("\(([^()]*)\)")


def _check_paren_match(text: str):
    left = text.count('(')
    right = text.count(')')
    
    if(left == right):
        return False
    
    return True


def _spelling_rep(text: str):
    
    is_error = _check_paren_match(text)
    if is_error:
        return None
    
    is_spelling = bool(randint(0,1))
    result = ""
    
    segment_list = SLASH_SEPARATED_PARENS.split(text)
    if(len(segment_list) != 1):
        for segment in segment_list:
            curr = INSIDE_PARANS.findall(segment)
            if(((len(curr) == 0) or (len(curr) % 2 == 1))):
                result += INSIDE_PARANS.sub("", segment)
            else:
                try:
                    if(bool(re.search('\d', curr[0])) and is_spelling):
                        result += curr[1]
                    else:
                        result += curr[0]
                except:
                    print(curr)
                    return None
        text = result
    
    return text


def _check_parse_error(text: str):
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


def etri_normalize(text: str):
    '''
        1. Parantheses inside parantheses check
        2. Check for incorrect paranthese count match
        3. 
    '''

    error = _check_parse_error(text)
    if error:
        return None

    modified_text = _spelling_rep(text)
    if modified_text is not None:
        text = modified_text
    else:
        return None

    text = ETRI_SPECIAL_SYMBOLS.sub("", text)
    text = SPACES.sub(" ", text).strip()

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