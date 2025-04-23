{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-1.5,1.5], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-1.5,1.5], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=0.5 and x<=1.0",
  "beta": "z>=1.5 and z<=2",
  "eta": "y>=-1.0 and y<=-0.5",
  "objectives": {
    "objective": "z"
  }
}

