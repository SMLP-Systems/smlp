{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[ -0.5,0.2], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[ -0.5,0.8], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=0.8 and x>=-0.8",
  "beta": "z>=1.5 and z<=1.1",
  "eta": "y>=-1 and y<=1.5",
  "objectives": {
    "objective": "z"
  }
}

