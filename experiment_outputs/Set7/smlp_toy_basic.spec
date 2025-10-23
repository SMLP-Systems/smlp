{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-3,-1], "rad-abs":0.1},
	{"label":"y", "interface":"knob", "type":"real", "range":[-4,-2], "rad-rel":0.1},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=-3 and x<=-1",
  "beta": "z>=2.6 and z<=3.14",
  "eta": "y>=-4 and y<=-2",
  "objectives": {
    "objective": "z"
  }
}

