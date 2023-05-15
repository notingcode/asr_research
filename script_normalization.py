import re

SOLUGATE_SPECIAL_SYMBOLS = re.compile("[\*\+/blon\.\?]")
SPACES = re.compile(r"\s+")
SLASH_SEPARATED_PARENS = re.compile("(\([^\)]*\)/\([^\)]*\))")
INSIDE_PARANS = re.compile("\(([^()]*)\)")
DIQUEST_SPECIAL_SYMBOLS = re.compile(r"[*FNOPS:]")

def cleanup_transcript(line_of_text):
    
    cond = check_parentheses_parsing(line_of_text)
    if cond == True:
        return None

    left = line_of_text.count('(')
    right = line_of_text.count(')')
    
    if(left != right or left%2 == 1):
        return None

    line_of_text = spelling_rep(line_of_text)
    
    if(bool(re.search('\d', line_of_text))):
        return None
    
    if('(' in line_of_text or ')' in line_of_text):
        return None

    line_of_text = SOLUGATE_SPECIAL_SYMBOLS.sub("", line_of_text)
    line_of_text = SPACES.sub(" ", line_of_text).strip()
    
    left = line_of_text.count('(')
    right = line_of_text.count(')')
    
    if(left != right or left%2 == 1):
        return None
    
    if(len(line_of_text) < 5):
        return None
    
    return line_of_text


def spelling_rep(text):
    result = ""
    
    segment_list = SLASH_SEPARATED_PARENS.split(text)
    if(len(segment_list) != 1):
        for segment in segment_list:
            curr = INSIDE_PARANS.findall(segment)
            if(len(curr) == 0):
                result += segment
            else:
                if(bool(re.search('\d', curr[0]))):
                    result += curr[1]
                else:
                    result += curr[0]
        return result
    
    return text

def check_parentheses_parsing(x):
    cond = False
    partitions = SLASH_SEPARATED_PARENS.split(x)
    if(len(partitions) != 1):
        for part in partitions:
            curr = INSIDE_PARANS.findall(part)
            for element in curr:
                if(('(' in element) or (')' in element)):
                    cond = True
                    break
    return cond

def edit_annotation(text):
    if INSIDE_PARANS.search(text) is not None:
        partition_list = INSIDE_PARANS.split(text)
        for partition in partition_list:
            if "FP" not in partition:
                partition = partition.split(":",1)[-1]
            
        text = "".join(partition_list)
        
    text = DIQUEST_SPECIAL_SYMBOLS.sub("",text)
    text = SPACES.sub(" ",text).strip()
    
    return text