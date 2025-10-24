{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[0.5,0.9], "rad-abs":0.01},
	{"label":"y", "interface":"knob", "type":"real", "range":[0.1,1], "rad-abs":0.01},
    {"label":"z", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=0.5 and x<=1",
  "beta": "z>=0.3",
  "eta": "y<1.5",
  "objectives": {
    "objective": "z"
  }
}

