import csv, json, os, sys, urllib.parse
sys.path.insert(0, os.path.dirname(__file__))
from bom_data import SECTIONS, ELECTRONICS, A, S

here = os.path.dirname(os.path.abspath(__file__))
rows = []
for sec, items in SECTIONS + [ELECTRONICS]:
    for it in items:
        r = dict(it)
        r["section"] = sec
        r["electronics"] = sec == ELECTRONICS[0]
        r["url"] = A + it["asin"] if it.get("asin") else None
        r["search"] = S + urllib.parse.quote_plus(it["q"]) if it.get("q") else None
        r["line"] = round(it["qty"] * it["price"], 2)
        rows.append(r)
with open(os.path.join(here, "bom_priced.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Section", "Part", "Spec", "Need", "Buy", "Packs", "Est. price each (USD)", "Est. line (USD)", "Amazon", "Notes"])
    for r in rows:
        link = r["url"] or (r.get("alt") or ("", ""))[1]
        w.writerow([r["section"], r["part"], r["spec"], r["need"], r["buy"], r["qty"], f'{r["price"]:.2f}',
                    f'{r["line"]:.2f}', link, r.get("note", "")])
json.dump(rows, open(os.path.join(here, "bom_rows.json"), "w"), indent=1, ensure_ascii=False)
mech = sum(r["line"] for r in rows if not r["electronics"])
elec = sum(r["line"] for r in rows if r["electronics"])
print(len(rows), "rows; mechanical", mech, "electronics", elec, "total", mech + elec)
tpl = open(os.path.join(here, "page_template.html")).read()
open(os.path.join(here, "capstan_arm_parts.html"), "w").write(tpl.replace("__ROWS__", json.dumps(rows, ensure_ascii=False)))
print("wrote capstan_arm_parts.html and bom_priced.csv")
