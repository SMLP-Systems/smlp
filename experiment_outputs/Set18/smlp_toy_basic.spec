{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[ 1.5,2], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[ 1.5,2], "rad-abs":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=2 and x>=1.1",
  "beta": "z>=1.5 and z<=2.1",
  "eta": "y>=1.5 and y<=1.7",
  "objectives": {
    "objective": "z"
  }
}

