
import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data
import yfinance as yf
from datetime import datetime
yf.pdr_override()
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import risk_matrix
from pypfopt.risk_models import sample_cov
import matplotlib.pyplot as plt

from pypfopt import plotting
import matplotlib.pyplot as plt
from pypfopt.efficient_frontier import EfficientFrontier
import pickle
from streamlit_tags import st_tags, st_tags_sidebar
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

@st.cache_data()
def name_tickers():
    with open('lista.pkl', 'rb') as f:
        lista_caricata = pickle.load(f)
    return lista_caricata

def aggiungi_stock(Stock, Amount):
    # Aggiunta di una nuova riga al DataFrame
    nuova_riga = pd.DataFrame({'Stock': [Stock], 'Amount': [Amount]})
    st.session_state.df = pd.concat([st.session_state.df, nuova_riga], ignore_index=True)


def pesi_actual_new(importi_investiti, margine_disponibile):
    totale_investito = importi_investiti.sum()
    pesi_actual = (importi_investiti)/totale_investito
    pesi_nuovi = importi_investiti/(totale_investito+margine_disponibile)
    return pesi_actual, pesi_nuovi

def estrai_stock(actual_tickers, new_tickers, start_date=datetime(2010, 2, 12), end_date = datetime.now()):
    tickers = actual_tickers + new_tickers
    total_tickers = len(tickers)
    matrice_prezzi = pd.DataFrame()
    for t in tickers:
        matrice_prezzi[t] = data.get_data_yahoo(t, start_date, end_date)['Adj Close']
    return matrice_prezzi

def mu_S(matrice_prezzi):
    global rendimenti_stocks, volatilita_stocks
    mu = mean_historical_return(matrice_prezzi, returns_data=False, compounding=False, frequency=252, log_returns=False)
    S = risk_matrix(matrice_prezzi, method='sample_cov')
    return mu, S

def random_w(actual_tickers, new_tickers, pesi_nuovi, n_samples = 10000):
    n_actual_tickers = len(actual_tickers)
    tickers = actual_tickers + new_tickers
    total_tickers = len(tickers)
    # Generazione di valori casuali normalizzati negli ultimi 12 elementi di ogni array
    random_values = np.random.rand(n_samples, total_tickers - n_actual_tickers)
    normalized_values = random_values * (1-pesi_nuovi.sum()) / np.sum(random_values, axis=1, keepdims=True)

    # Ripetizione dell'array di base per il numero di array desiderato
    pesi_nuovi_repeated = np.tile(pesi_nuovi, (n_samples, 1))

    # Combinazione dei valori fissi e quelli normalizzati
    w = np.column_stack((pesi_nuovi_repeated, normalized_values))
    return w

def plotting_ef(mu, S, actual_tickers, new_tickers, pesi_actual, pesi_nuovi, w):
    from pypfopt.efficient_frontier import EfficientFrontier
    n_actual_tickers = len(actual_tickers)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef.add_constraint(lambda x : x[:n_actual_tickers] >= pesi_nuovi)#assegno i pesi veri
    # ef.add_constraint(lambda x : x[11] <= 0.25)
    fig, ax = plt.subplots()

    ef_max_sharpe = ef.deepcopy()
    ef_min_volatility = ef.deepcopy()
    ef_efficient_risk = ef.deepcopy()
    ef_efficient_return = ef.deepcopy()

    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
    
    
    # Generate random portfolios
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds # !!! porque no resta la taso libre de riesgo?
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Trovo il portafoglio con sharpe massimo. il calcolo usa la tasa libre de riesgo
    weights_max_sharpe = ef_max_sharpe.max_sharpe()
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Trovo il portafoglio con la minore volatilità
    weights_min_volatility = ef_min_volatility.min_volatility()
    ret_tangent, std_tangent, _  = ef_min_volatility.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="g", label="Min Volatility")
    
    

    # Portafoglio attuale
    ef = EfficientFrontier(mu[actual_tickers], S.loc[actual_tickers,actual_tickers], weight_bounds=(0, 1))
    ef.add_constraint(lambda x: x == pesi_actual)#assegno i pesi veri
    ef_actual = ef.deepcopy()
    weights_actual = ef_actual.max_sharpe()#trovo il max_sharpe per i pesi attuali.
    ret_tangent, std_tangent, _ = ef_actual.portfolio_performance() #le performance coincidono con rendimenti_stocks e volatilita_stocks
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="b", label="Actual")

    # Output
    ax.set_title("Efficient Frontier with random portfolios")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ef_scatter.png", dpi=200)
    
    plt.show()

    #st.pyplot(fig) # IMPORTANTE PER STREAMLIT (Meglio printare la immagine salva)

    return ef_max_sharpe, weights_max_sharpe, ef_actual, weights_actual


def allocazione(matrice_prezzi, weights_max_sharpe, total_portfolio_value):

    latest_prices = get_latest_prices(matrice_prezzi)
    da = DiscreteAllocation(weights_max_sharpe, latest_prices, total_portfolio_value)
    allocation, leftover = da.lp_portfolio()
    df_allocation = pd.DataFrame(list(allocation.items()), columns=["Stock", "Quantità"])
    return df_allocation



