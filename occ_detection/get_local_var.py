import __main__
def whos(lss):
    fil = __main__.__file__
    thisthinghere = open(fil).read().split("\n")
    vs = []
    for l in thisthinghere:
        if l.find("=") > -1:
            vs.append(l.split("=")[0].strip())
    keys = lss.keys()
    out = {}
    for v in vs:
        try: 
            out[v] = lss[v]
        except:
            "not in list"
    keys = out.keys()
    keys.sort()
    for k in keys:
        val = str(out[k])
        if len (val) > 10:
            if val[-1] == ")":val = val[0:10]+"..."+val[-10:]
            elif val[-1] == "]" :val = val[0:10]+"..."+val[-10:]
            else: val = val[0:10]
        print k,":",val

    return out
