{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-0.5,1], "rad-abs":0.1},
    {"label":"y", "interface":"output", "type":"real"}
  ],
  "alpha": "x<=1.7 and x>=-0.1",
  "beta": "y<=2.5 and y>=1.5",
  "eta": "x>=-0.1 and x<=0.3",
  "objectives": {
    "objective": "y"
  }
}

