{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-0.5,1.3], "rad-abs":0.1},
    {"label":"y", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=5 and x>= 3.5",
  "beta": "y<=25 and y>=22.5",
  "eta": "x>=3.7 and x<=4.5",
  "objectives": {
    "objective": "y"
  }
}

