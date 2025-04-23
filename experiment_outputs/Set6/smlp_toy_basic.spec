{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2,2], "rad-abs":0.01},
	{"label":"y", "interface":"knob", "type":"real", "range":[-2,2], "rad-rel":0.01},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-2 and x<=-0.5",
  "beta": "z<=6 and z>=4.5",
  "eta": "y<=2 and y>=-2",
  "objectives": {
    "objective": "z"
  }
}

