# 🚗 Car Price Predictor — Streamlit App

A Streamlit web app that predicts the resale price of a used car using a
Linear Regression model, built from the analysis in `maincode.ipynb`.

## Project structure

```
car_price_predictor/
├── app.py                # Streamlit web app
├── train_model.py        # Data cleaning + model training (reproduces the notebook)
├── Linearmodelcar.pkl    # Pre-trained model (ready to use)
├── requirements.txt      # Python dependencies
├── data/
│   ├── car.csv            # Raw dataset
│   └── cleaned_car.csv    # Cleaned dataset
└── README.md
```

## How to run it in PyCharm

1. **Open the folder**
   `File → Open...` and select the `car_price_predictor` folder.

2. **Create a virtual environment** (PyCharm usually prompts you to do this
   automatically when it detects `requirements.txt`; otherwise):
   - `File → Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv Environment`

3. **Install dependencies.** Open the PyCharm terminal (bottom toolbar) and run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app.** In the same terminal:
   ```bash
   streamlit run app.py
   ```
   PyCharm will also let you right-click `app.py` and add a run
   configuration, but Streamlit apps are launched via the `streamlit run`
   CLI command, not the normal "Run" button, so the terminal is the easiest
   way.

5. Your browser will open automatically at **http://localhost:8501**.

## Retraining the model

The included `Linearmodelcar.pkl` was trained with the exact package
versions in `requirements.txt`. If you ever change the dataset or upgrade
scikit-learn, regenerate it with:

```bash
python train_model.py
```

This re-runs the same cleaning steps and modeling approach as the original
notebook (including the 1000-iteration search for the best `random_state`
train/test split) and overwrites `Linearmodelcar.pkl` and
`data/cleaned_car.csv`.

## App features

- **Predict Price tab** — pick a company, model, year, fuel type, and
  kilometers driven, then get an instant price estimate.
- **Explore Data tab** — interactive charts: average price by company,
  price vs. kilometers driven, price distribution by year, fuel type split,
  plus a browsable data table.
- **About tab** — explains the model and project structure.

## Troubleshooting

- **"Could not unpickle model" / version errors** — run
  `python train_model.py` to regenerate `Linearmodelcar.pkl` with your
  currently installed scikit-learn version.
- **Port already in use** — run `streamlit run app.py --server.port 8502`.
