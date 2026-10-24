import glob, json, sys, collections
rows = collections.defaultdict(list)
for f in sorted(glob.glob(sys.argv[1])):
    for l in open(f):
        r = json.loads(l)
        rows[(r["task"], r["ctx"], r["fmt"], r["cond"])].append(r)
print(f"{'task':<7}{'ctx':>6} {'fmt':<7}{'cond':<44}{'n':>4}{'acc':>7}{'f1':>7}{'ans_lp':>9}")
for k, v in sorted(rows.items()):
    n = len(v)
    print(f"{k[0]:<7}{k[1]:>6} {k[2]:<7}{k[3]:<44}{n:>4}{sum(r['acc'] for r in v)/n:7.3f}{sum(r['f1'] for r in v)/n:7.3f}{sum(r['ans_lp'] for r in v)/n:9.3f}")
