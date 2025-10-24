{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-1.5,0.5], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-0.3,0.3], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=0 and x>=-1.5",
  "beta": "z>=1.5 and z<=2.5",
  "eta": "y>=-1 and y<=0.5",
  "objectives": {
    "objective": "z"
  }
}

