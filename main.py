import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from sklearn.metrics import root_mean_squared_error
import warnings

warnings.filterwarnings("ignore")  # Ascundem avertismentele estetice

#date.csv -> datele reale lunare (2020 - 2026) pentru cursul mediu BNR,
# Inflatia anualizata (INS) si ROBOR la 3 luni.

# 0. PREGATIREA DATELOR
# Citim fisierul CSV creat de noi
df = pd.read_csv('date.csv', index_col='Data', parse_dates=True)

# Setam frecventa pe final de luna (ME = Month End)
df.index.freq = 'ME'

# ==========================================
# PART 1: ANALIZA UNIVARIATA (Curs_EUR)
# ==========================================
print("--- 1. ANALIZA UNIVARIATA ---")
serie_eur = df['Curs_EUR']

# 1.1 Test Stationaritate (ADF)
adf_result = adfuller(serie_eur)
print(f"Test ADF (Nivel): p-value = {adf_result[1]:.4f} -> {'Nestationara' if adf_result[1] > 0.05 else 'Stationara'}")
if adf_result[1] > 0.05:
    adf_diff = adfuller(serie_eur.diff().dropna())
    print(f"Test ADF (Diferentiata): p-value = {adf_diff[1]:.4f} -> Seriile sunt integrate.")

# 1.2 Split Training (80%) / Test (20%)
train_size = int(len(serie_eur) * 0.8)
train, test = serie_eur.iloc[:train_size], serie_eur.iloc[train_size:]

# 1.3 Holt-Winters (Netezire Exponentiala)
hw_model = ExponentialSmoothing(train, trend='add', seasonal=None, initialization_method="estimated").fit()
hw_pred = hw_model.forecast(len(test))

# 1.4 SARIMA
sarima_model = SARIMAX(train, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
sarima_pred = sarima_model.get_forecast(steps=len(test))
sarima_mean = sarima_pred.predicted_mean
sarima_ci = sarima_pred.conf_int()

# 1.5 Compararea acuratetei
rmse_hw = root_mean_squared_error(test, hw_pred)
rmse_sarima = root_mean_squared_error(test, sarima_mean)

print(f"Acuratete (RMSE) - Holt-Winters: {rmse_hw:.4f}")
print(f"Acuratete (RMSE) - SARIMA: {rmse_sarima:.4f}")

# Plot Univariate
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test (Real)')
plt.plot(test.index, hw_pred, label='Prognoza Holt-Winters', linestyle='--')
plt.plot(test.index, sarima_mean, label='Prognoza SARIMA', linestyle='--')
plt.fill_between(test.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3,
                 label='Interval Incredere SARIMA')
plt.title('Prognoza Univariata: Curs RON/EUR')
plt.legend()
plt.grid(True)

plt.savefig('01_Prognoza_Univariata.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# PART 2: ANALIZA MULTIVARIATA
# ==========================================
print("\n--- 2. ANALIZA MULTIVARIATA ---")

# 2.1 Cointegrare (Johansen)
johansen_test = coint_johansen(df, det_order=0, k_ar_diff=1)
print("Test Johansen (Statistica Trace):", johansen_test.lr1)
print("Valori critice (95%):", johansen_test.cvt[:, 1])

# 2.2 Model VAR (pe serii diferentiate)
df_diff = df.diff().dropna()
var_model = VAR(df_diff)
var_results = var_model.fit(maxlags=1, ic='aic') # setat la maxlags=1 pentru setul nostru limitat de date reale
print("\nOrdinul optim de lag (selectat de AIC):", var_results.k_ar)

# 2.3 Cauzalitate Granger
print("\nTest Granger: Cauzeaza Inflatia -> Cursul_EUR?")
granger_res = grangercausalitytests(df_diff[['Curs_EUR', 'Inflatie']], maxlag=1, verbose=False)
p_val = granger_res[1][0]['ssr_ftest'][1]
print(f"Lag 1: p-value = {p_val:.4f}")

# 2.4 Functia de Raspuns la Impuls (IRF)
irf = var_results.irf(5)
fig_irf = irf.plot(response='Curs_EUR', impulse='ROBOR')
fig_irf.suptitle('Raspunsul Cursului EUR la un soc in ROBOR', fontsize=12)
fig_irf.tight_layout()
plt.savefig('02_Impuls_IRF.png', dpi=300, bbox_inches='tight')
plt.show()

# 2.5 Descompunerea Variantei (FEVD)
fevd = var_results.fevd(5)
fig_fevd = fevd.plot()
fig_fevd.suptitle('Descompunerea Variantei (FEVD)', fontsize=12)
fig_fevd.tight_layout()
plt.savefig('03_Descompunere_Varianta_FEVD.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nGata!")
