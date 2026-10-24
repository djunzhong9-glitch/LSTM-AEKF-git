"""Generate an editable draw.io version of Figure 2 in Figure 1's layout."""

from pathlib import Path
from xml.etree import ElementTree as ET


OUT = Path(__file__).with_name("fig2_fig1_style.drawio")
mxfile = ET.Element("mxfile", host="app.diagrams.net", modified="2026-10-23T00:00:00.000Z", agent="Codex", version="24.7.17", type="device")
diagram = ET.SubElement(mxfile, "diagram", id="fig2-as-fig1", name="Figure 2 — Figure 1 style")
model = ET.SubElement(diagram, "mxGraphModel", dx="2500", dy="1300", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="2500", pageHeight="1300", math="0", shadow="0")
root = ET.SubElement(model, "root")
ET.SubElement(root, "mxCell", id="0")
ET.SubElement(root, "mxCell", id="1", parent="0")

NAVY = "#1E293B"
BLUE = "#3974A5"
PALE_BLUE = "#EDF4FA"
ORANGE = "#C96B2B"
PALE_ORANGE = "#FFF3EA"
TEAL = "#008E98"
PALE_TEAL = "#EAF7F6"
GRAY = "#5E6874"


def vertex(id, label, x, y, w, h, *, fill="none", stroke="none", color=NAVY,
           size=23, bold=False, align="center", rounded=False, extra=""):
    style = (f"rounded={int(rounded)};whiteSpace=wrap;html=1;fillColor={fill};"
             f"strokeColor={stroke};fontColor={color};fontSize={size};"
             f"fontFamily=Georgia;fontStyle={1 if bold else 0};align={align};"
             f"verticalAlign=middle;spacing=8;{extra}")
    cell = ET.SubElement(root, "mxCell", id=id, value=label, style=style, vertex="1", parent="1")
    ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
    return id


def edge(id, source, target, *, color=NAVY, dashed=False, points=None, label="", width=3, extra=""):
    style = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
             f"html=1;strokeColor={color};strokeWidth={width};endArrow=block;"
             f"endFill=1;dashed={int(dashed)};fontColor={color};fontSize=18;{extra}")
    cell = ET.SubElement(root, "mxCell", id=id, value=label, style=style, edge="1", parent="1", source=source, target=target)
    geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
    if points:
        arr = ET.SubElement(geo, "Array", **{"as": "points"})
        for x, y in points:
            ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))
    return id


# Figure 1's upper design sequence.
vertex("top_title", "Offline model design", 28, 18, 500, 52, color=BLUE, size=35, bold=True, align="left")
vertex("top_note", "before online filtering", 530, 20, 540, 50, size=29, align="left")
vertex("top_right", "GM-9D + adaptive / shadow EKF + LSTM", 1780, 20, 690, 50, color=GRAY, size=25, align="right")
vertex("line1", "", 28, 78, 2440, 2, fill="#CDD9E4")

heads = [
    ("h1", "Ground model", 30),
    ("h2", "9D discretization", 520),
    ("h3", "Dual-filter design", 1010),
    ("h4", "Feature pipeline", 1500),
    ("h5", "LSTM outputs", 1990),
]
for id, label, x in heads:
    vertex(id, label, x, 95, 450, 52, size=30, bold=True)

vertex("s1", "Measurements zₖ<br>(position / velocity)<br><br>xᴳᴹ = [rᵀ, vᵀ, a<sub>g</sub>ᵀ]ᵀ ∈ ℝ⁹", 45, 158, 420, 132,
       fill=PALE_BLUE, stroke="#94AEC7", size=25)
vertex("s2", "Singer acceleration process<br>Φᴳᴹ, Qᴳᴹ via Van Loan<br>P<sub>aa,0</sub> = q<sub>g</sub>τ<sub>g</sub>³ / 2", 535, 158, 420, 132,
       fill=PALE_BLUE, stroke="#94AEC7", size=23)
vertex("s3", "Main: adaptive Q / R<br>Shadow: fixed Q₀ / R₀<br>independent recursion", 1025, 158, 420, 132,
       fill=PALE_BLUE, stroke="#94AEC7", size=24)
vertex("s4", "Innovation, P, Jacobian, …<br>15-dimensional φₖ<br>standardize &amp; clip", 1515, 158, 420, 132,
       fill=PALE_BLUE, stroke="#94AEC7", size=24)
vertex("s5", "âₖ, ΔQ̂ₖ, ΔR̂ₖ<br>acceleration / noise updates<br>trained on fixed-EKF features", 2005, 158, 420, 132,
       fill=PALE_TEAL, stroke=TEAL, size=24)
for n in range(1, 5):
    edge(f"step{n}", f"s{n}", f"s{n+1}", width=3)

vertex("line2", "", 28, 325, 2440, 2, fill="#CDD9E4")
vertex("template_title", "Prepared template", 28, 342, 550, 56, color=ORANGE, size=35, bold=True, align="left")
vertex("template", "GM-9D dynamics: Φᴳᴹ, Qᴳᴹ   │   measurement H: [r, v]   │   shadow Q₀ / R₀   │   feature normalizer   │   trained LSTM   │   clip bounds", 40, 411, 2385, 88,
       fill=PALE_ORANGE, stroke="#DCA57E", size=27)
edge("offline_to_template", "s5", "template", color=TEAL, width=2, extra="exitX=0.5;exitY=1;entryX=0.9;entryY=0;")

# Figure 1's lower service path, with the two filters and GM-9D fusion retained.
vertex("service_title", "Online filtering", 28, 525, 550, 54, color=TEAL, size=37, bold=True, align="left")
vertex("measure", "zₖ<br>position / velocity", 42, 660, 195, 110,
       fill=PALE_BLUE, stroke="#94AEC7", size=25)
vertex("main", "Main adaptive EKF<br>adaptive Q / R<br>+ acceleration correction", 300, 598, 270, 130,
       fill=PALE_ORANGE, stroke="#E4A083", size=23)
vertex("shadow", "Shadow fixed EKF<br>fixed Q₀ / R₀<br>feature generation only", 300, 788, 270, 130,
       fill="#EAF6E9", stroke="#8FBE91", size=23)
vertex("main_diag", "Main diagnostics<br>innovation, P, Jacobian, …", 630, 605, 295, 115,
       fill=PALE_ORANGE, stroke="#E4A083", size=22)
vertex("shadow_diag", "Shadow diagnostics<br>innovation, P, Jacobian, …", 630, 792, 295, 115,
       fill="#EAF6E9", stroke="#8FBE91", size=22)
vertex("feat", "15D feature φₖ", 985, 650, 248, 92,
       fill=PALE_BLUE, stroke="#80AFE0", size=26, bold=True)
vertex("norm", "Standardize<br>&amp; clip", 1270, 650, 230, 92,
       fill=PALE_BLUE, stroke="#80AFE0", size=25)
vertex("lstm", "LSTM<br>trained on fixed-EKF features", 1540, 639, 306, 116,
       fill="#F1EDFF", stroke="#A99ADE", size=24, bold=True)
vertex("outputs", "âₖ  acceleration correction<br>ΔQ̂ₖ  process noise update<br>ΔR̂ₖ  measurement noise update", 1900, 612, 475, 165,
       fill=PALE_TEAL, stroke=TEAL, size=23)

edge("m_to_main", "measure", "main", extra="exitX=1;exitY=0.25;entryX=0;entryY=0.5;")
edge("m_to_shadow", "measure", "shadow", extra="exitX=1;exitY=0.75;entryX=0;entryY=0.5;")
edge("main_d", "main", "main_diag")
edge("shadow_d", "shadow", "shadow_diag")
edge("d1_feat", "main_diag", "feat", extra="exitX=1;exitY=0.5;entryX=0;entryY=0.25;")
edge("d2_feat", "shadow_diag", "feat", extra="exitX=1;exitY=0.5;entryX=0;entryY=0.75;")
edge("feat_norm", "feat", "norm")
edge("norm_lstm", "norm", "lstm")
edge("lstm_out", "lstm", "outputs")

vertex("line3", "", 28, 962, 2440, 2, fill="#CDD9E4")
vertex("fusion_title", "GM-9D residual fusion", 28, 977, 720, 54, color=ORANGE, size=35, bold=True, align="left")
vertex("shadow_accel", "Shadow acceleration<br>â<sub>g,k|k</sub><sup>sh</sup>", 320, 1060, 245, 94,
       fill="#EAF6E9", stroke="#8FBE91", size=23)
vertex("residual", "Residual correction<br>δâₖ = clip[gₐ ⊙ (âₖᴸ − â<sub>g,k|k</sub><sup>sh</sup>), ±a<sub>max</sub>]", 760, 1050, 520, 112,
       fill=PALE_ORANGE, stroke="#E4A083", size=22)
vertex("gm", "Main GM-9D EKF<br>x<sub>k+1|k</sub> = Φᴳᴹ x<sub>k|k</sub> + G δâₖ<br>P<sub>k+1|k</sub> = Φᴳᴹ P<sub>k|k</sub> Φᴳᴹᵀ + Qᴳᴹ", 1450, 1035, 560, 140,
       fill=PALE_BLUE, stroke="#80AFE0", size=22)
vertex("state", "State estimate<br>[r, v, a<sub>g</sub>] (9D)<br>NEES / 6 using [r, v]", 2150, 1035, 300, 140,
       fill=PALE_TEAL, stroke=TEAL, size=23)
edge("shadow_acc", "shadow", "shadow_accel", color=BLUE, points=[(360, 1000)], extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
edge("shadow_res", "shadow_accel", "residual")
edge("out_res", "outputs", "residual", color=ORANGE, points=[(2070, 990), (1020, 990)], extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
edge("res_gm", "residual", "gm", color=ORANGE)
edge("gm_state", "gm", "state", color=TEAL)
edge("out_gm", "outputs", "gm", color=BLUE, label="form Qᴳᴹ + ΔQ̂ₖ; Rₖ₊₁ from ΔR̂ₖ", points=[(2110, 960), (1730, 960)], extra="exitX=0.5;exitY=1;entryX=0.5;entryY=0;")
vertex("footnote", "âₖ feeds the next prediction step (k → k + 1); the shadow filter remains fixed for consistent features.", 90, 1212, 2300, 46,
       color=GRAY, size=22, align="center")

tree = ET.ElementTree(mxfile)
ET.indent(tree, space="  ")
tree.write(OUT, encoding="utf-8", xml_declaration=True)
print(OUT)
