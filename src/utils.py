def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.instersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
