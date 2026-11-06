"""Replace Chapter 3 with _ch3_new.html and renumber the equation references elsewhere (old 3.3-3.6 -> 3.12, 3.20-3.22)."""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "report.html"
s = open(p, encoding="utf-8").read()
if 'id="s3-13"' in s:
    print("already patched"); sys.exit(0)
new = open("_ch3_new.html", encoding="utf-8").read().strip()
m = re.search(r'<section class="sheet" id="ch3">.*?</section>', s, flags=re.S)
assert m
before, after = s[:m.start()], s[m.end():]
mp = {"3": "3.12", "4": "3.20", "5": "3.21", "6": "3.22"}
def fix(t):
    return re.sub(r"Equation \(3\.([3-6])\)", lambda x: "Equation (%s)" % mp[x.group(1)], t)
s = fix(before) + new + fix(after)
open(p, "w", encoding="utf-8").write(s)
print("patched chapter 3")
