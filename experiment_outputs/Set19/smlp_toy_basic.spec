{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-0.5,0.2], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-0.5,0.2], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=1 and x>=-0.1",
  "beta": "z>=-0.5 and z<=0.1",
  "eta": "y>=-0.5 and y<=0.7",
  "objectives": {
    "objective": "z"
  }
}

