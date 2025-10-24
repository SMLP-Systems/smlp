{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[1,1.5], "rad-abs":0.1},
    {"label":"y", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=1 and x<=1.5",
  "beta": "y>=0 and y<=1.5",
  "eta": "x>=1.2 and x<=1.6",
  "objectives": {
    "objective": "y"
  }
}

