{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[0,1.7], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[0,1.7], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=1.7 and x>=1.1",
  "beta": "z>=0.5 and z<=1.5",
  "eta": "y>=1.1 and y<=1.7",
  "objectives": {
    "objective": "z"
  }
}

