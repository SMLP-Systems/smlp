{
  "version": "1.2",
  "variables": [
    {"label":"x", "interface":"knob", "type":"real", "range":[-2.0,0], "rad-abs":0.1},
    {"label":"y", "interface":"output", "type":"real"}
  ],
  "alpha": "x>=4.5 and x<=7.5",
  "beta": "y>=0 and y<=2",
  "eta": "x>=6.0 and x<=7",
  "objectives": {
    "objective": "y"
  }
}

