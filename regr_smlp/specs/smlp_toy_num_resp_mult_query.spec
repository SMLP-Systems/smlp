{
    "version": "1.1",
    "spec":[
	{"label": "y1", "type": "response", "range": "float"},
	{"label": "y2", "type": "response", "range": "float"},
	{"label": "x", "type": "input", "range": "float", "bounds": [0,10]},
	{"label": "p1", "type": "knob", "range": "float", "rad-rel": 0.1, "grid": [2,4,7], "bounds": [0,10]},
	{"label": "p2", "type": "knob", "range": "float", "rad-abs": 0.2, "bounds": [3,7]}],
    "queries": {"query1": "(y2**3+p2)/2<6","query2": "y1>=9","query3": "y2<0"}
}
