# ASTRA-SIFT — Streamlit Demo App

Risk-aware satellite information selection under a limited communication budget.

This is a **synthetic, proof-of-concept demonstration**. It does not use real
satellite imagery, real satellite hardware, or a real communication link. It
does not change, retrain, or re-tune the underlying ASTRA-SIFT selection
logic in any way — this app only adds a web UI on top of the exact same
functions and the exact same trained-policy training call already used
elsewhere in the project.

## Folder contents

```
astra_sift_app/
├── app.py                          <- the Streamlit application
├── requirements.txt                <- Python dependencies
├── README.md                       <- this file
└── results/
    └── figures/
        ├── catastrophe_recall.png  <- from the full 200-episode evaluation
        ├── mean_reward.png         <- from the full 200-episode evaluation
        └── regret_vs_oracle.png    <- from the full 200-episode evaluation
```

The three PNGs under `results/figures/` are the **actual saved figures** from
the previously reported in-distribution / out-of-distribution evaluation —
they are bundled here so the app can display them, but they are not
regenerated or recomputed by this app.

## Running it locally

1. **Install Python 3.9+** if you don't already have it.

2. **(Recommended) create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**, from inside the `astra_sift_app/` folder (so the
   `results/figures/` path resolves correctly):
   ```bash
   streamlit run app.py
   ```

5. Streamlit will print a local URL, typically:
   ```
   Local URL: http://localhost:8501
   ```
   Open that in your browser.

6. Click **"Run Satellite Pass Demo"** to generate the one simulated
   satellite pass, see what Top-K importance vs. ASTRA-SIFT would each
   transmit, whether the critical observation was preserved, the comparison
   chart, and (if present) the three final evaluation figures.

## Notes

- The "trained policy" used here is produced by the exact same training call
  (same seed, same hyperparameters) as the one reported in the final
  experiment — nothing is retrained or re-tuned inside this app.
- The 15-observation demo pass is a fixed, hand-built illustration, not a new
  experiment. The actual measured recall/regret numbers come from the
  200-episode evaluation figures shown at the bottom of the page.
- If `results/figures/` is missing or empty, the app still runs — it will
  just show a note that those figures weren't found instead of an image.
