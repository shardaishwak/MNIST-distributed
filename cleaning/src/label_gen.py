import re
import os
from pathlib import Path

# Field index → Y and X layout (based on your LABELS_14 structure)
# Format: (Y, X_offset_index)
# X positions per row are fixed based on your example
ROW_LAYOUT = {
    # Y: [x1, x2, x3, ...]
    740:  [220, 870, 1530],                          # fld_3,4,5
    900:  [220, 455, 755, 1110, 1530],              # fld_6-10
    1060: [220, 520, 875, 1290, 2000],              # fld_11-15
    1250: [220, 575, 990, 1700, 1940],              # fld_16-20
    1375: [220, 640, 1345, 1585, 1880],             # fld_21-25
    1535: [220, 930, 1170, 1465, 1820],             # fld_26-30
    1700: [220],                                     # fld_31
    1875: [220],                                     # fld_32
    2230: [220],                                     # fld_33
}

# Map field index to (Y, X list index)
FIELD_TO_Y_AND_SLOT = {}
field_idx = 3
for y, x_list in ROW_LAYOUT.items():
    for slot, _ in enumerate(x_list):
        if field_idx <= 33:
            FIELD_TO_Y_AND_SLOT[field_idx] = (y, slot)
            field_idx += 1

# Fixed sentence for fld_33 (prompt, not user input)
CONSTITUTION_PROMPT = (
    "We,thePeopleoftheUnitedStates,inordertoformamoreperfectUnion,"
    "establishJustice,insuredomesticTranquility,provideforthecommonDefense,"
    "promotethegeneralWelfare,andsecuretheBlessingsofLibertytoourselvesandourposterity,"
    "doordainandestablishthisCONSTITUTIONfortheUnitedStatesofAmerica."
)

def generate_label_dict_from_txt(txt_path):
    txt_path = Path(txt_path)
    match = re.search(r'_(\d{2})\.txt$', txt_path.name)
    if not match:
        raise ValueError(f"Filename must be like ref_XX.txt; got {txt_path.name}")
    suffix = match.group(1)

    # Read all field data
    field_values = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('fld_'):
                continue
            parts = line.split(maxsplit=1)
            field_name = parts[0]
            try:
                idx = int(field_name[4:])
            except ValueError:
                continue
            if idx < 3:  # Skip fld_0, fld_1, fld_2
                continue
            if idx > 33:
                continue
            value = parts[1] if len(parts) > 1 else ""
            field_values[idx] = value

    # Build nested dict: {y: {x: label}}
    label_dict = {}
    for idx in range(3, 34):
        if idx not in field_values:
            continue
        val = field_values[idx]
        # Special handling for fld_33: use fixed prompt if empty
        if idx == 33:
            if not val.strip():
                val = CONSTITUTION_PROMPT
            else:
                pass

        if not val:  # skip empty
            continue

        if idx not in FIELD_TO_Y_AND_SLOT:
            continue
        y, slot = FIELD_TO_Y_AND_SLOT[idx]
        x = ROW_LAYOUT[y][slot]

        if y not in label_dict:
            label_dict[y] = {}
        label_dict[y][x] = val

    return suffix, label_dict

def append_to_labels_py(suffix, label_dict, output_file="labels.py"):
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"LABELS_{suffix} = {{\n")
        for y in sorted(label_dict.keys()):
            inner = label_dict[y]
            f.write(f"    {y} : {{")
            items = []
            for x in sorted(inner.keys()):
                val = inner[x]
                # Escape single quotes for Python string
                if "'" in val and not '"' in val:
                    items.append(f'{x}: "{val}"')
                else:
                    val_escaped = val.replace("'", "\\'")
                    items.append(f"{x}: '{val_escaped}'")
            f.write(", ".join(items))
            f.write("},\n")
        f.write("}\n\n")

if __name__ == "__main__":
    txt_dir = "data/raw/truerefs"

    txt_lst = [
        os.path.join(txt_dir, f)
        for txt_dir, _, files in os.walk(txt_dir)
        for f in files
    ]

    for txt_file in txt_lst:
        suffix, label_dict = generate_label_dict_from_txt(txt_file)
        append_to_labels_py(suffix, label_dict)