#!/usr/bin/bash
sed -i.bak '/^from/iif os.path.exists("\/.dockerenv"): \
    print("Script is running inside a Docker container.") \
    import matplotlib \
    matplotlib.use("TkAgg") \
    import matplotlib.pyplot as plt\n' /app/smlp/src/run_smlp.py
