import json
import jsonlines

dir1 = "pred.json"
dir2 = "test.json"

with open(dir1, "r") as f:
    d1 = f.readlines()

with open(dir2, "r") as f:
    d2 = f.readlines()
all_pairs = []
for i1, i2 in zip(d1, d2):
    data_piece = {}
    i_dict1 = json.loads(i1.strip())
    i_dict2 = json.loads(i2.strip())
    if i_dict1["judge"] == "false":
        data_piece["prompt"] = i_dict1["source"]
        data_piece["reject"] = i_dict1["predict"]
        data_piece["accept"] = i_dict2["target"][0]
        all_pairs.append(data_piece)
with jsonlines.open("dpo.json", "w") as f:
    f.write_all(all_pairs)
print("done")
