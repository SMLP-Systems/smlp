{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2,2], "rad-rel":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.01},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-1 and x<=1",
  "beta": "z<=1.5 and z>=-1",
  "eta": "y>=-1 and y<=1",
  "objectives": {
    "objective": "z"
  }
}

