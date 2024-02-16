from codice_per_app import *
import streamlit as st
#CONFIGURACION DE LA PÁGINA
st.set_page_config(
     page_title = 'Portfolio Management',
     #page_icon = 'DS4B_Logo_Blanco_Vertical_FB.png',
     layout = 'wide')

lista_caricata = name_tickers()

with st.sidebar:

    st.title("Compila le informazioni del tuo portafoglio attuale:")

    # Creare il DataFrame
    portafoglio = [
        ('MMM', 2635.25),
        ('ACN', 1361.1),
        ('BABA', 339.62),
        ('CHK', 2601.27),
        ('MPW', 340.17),
        ('SRG.MI', 435.8),
        ('SNOW', 430.19),
        ('DIS', 101.59),
        ('EUNA.DE', 93.29),
        ('VWCE.DE', 3408.6)]

    df_portfolio = pd.DataFrame(portafoglio, columns=['Stock', 'Amount'])

    df_portfolio = st.data_editor(df_portfolio, use_container_width=True,
        column_config={
            "Stock": "Ticket Stock",
            "Amount": st.column_config.NumberColumn(
            "Importo Investito",
            help="Inserire l'importo investito",
            min_value=0,
            step=100,
            format="%d € ") },
        hide_index=True,
        num_rows="dynamic")

    # Seleziona nuovi stock da investire:
    new_tickers = st_tags_sidebar(
        label='# Inserisci gli stock di interesse:',
        text='Press enter to add more',
        suggestions=lista_caricata,
        maxtags = 5,
        key='2')

    margine_disponibile = st.number_input("Inserisci il totale da investire:", min_value=0, step=100)

    actual_tickers = list(df_portfolio.Stock)
    
    importi_investiti = np.array(df_portfolio.Amount)

    total_portfolio_value = importi_investiti.sum() + margine_disponibile

#MAIN
st.title('Ottimizzazione del portafoglio - Markowitz')

if st.sidebar.button('CALCOLARE PORTAFOGLIO', use_container_width=True):   
    # Calcoli interni
    pesi_actual, pesi_nuovi = pesi_actual_new(importi_investiti, margine_disponibile)

    matrice_prezzi = estrai_stock(actual_tickers, new_tickers)

    mu, S = mu_S(matrice_prezzi)
    
    w = random_w(actual_tickers, new_tickers, pesi_nuovi, n_samples = 10000)

    # Rendimento complessivo con pesi attuali
    rendimenti_stocks = (np.dot(mu[actual_tickers], pesi_actual))

    # Varianza e Volatilità complessiva del portafoglio attuale
    S_actual = S.loc[actual_tickers, actual_tickers]
    varianza_stocks = np.dot(pesi_actual.T, np.dot(S_actual, pesi_actual ))
    volatilita_stocks = np.sqrt(varianza_stocks)

    st.markdown(f"Il rendimento totale annualizzato è pari al {round(rendimenti_stocks * 100, 2)}%")
    st.markdown(f"La volatilità totale annualizzata è pari al {round(volatilita_stocks * 100, 2)}%")

    # Plotting
    ef_max_sharpe, weights_max_sharpe, ef_actual, weights_actual = plotting_ef(mu, S, actual_tickers, new_tickers, pesi_actual, pesi_nuovi, w)
    st.image("ef_scatter.png")

    # Pesi actual e pesi con Max Sharpe Ratio
    performance_actual = ef_actual.portfolio_performance(verbose=True)
    performance_new    = ef_max_sharpe.portfolio_performance(verbose=True)
    col1,col2 = st.columns(2)

    col1.markdown(f"Le perfomance del portafoglio attuale risulta avere un rendimento medio del **{round(performance_actual[0] * 100, 2)}%** e una variabilità del **{round(performance_actual[1] * 100, 2)}%**. L'indice Sharpe Ratio risulta pari a **{round(performance_actual[2], 2)}** ")
    col2.markdown(f"Le perfomance del portafoglio nuovo risulta avere un rendimento medio del **{round(performance_new[0] * 100, 2)}%** e una variabilità del **{round(performance_new[1] * 100, 2)}%**. L'indice Sharpe Ratio risulta pari a **{round(performance_new[2], 2)}** ")

    col1.markdown("Pesi Portfolio Actual:")
    somma_pesi_nuovi = pesi_nuovi.sum()
    weights_actual_rapportati = {chiave: valore * somma_pesi_nuovi for chiave, valore in weights_actual.items()}

    for chiave, valore in weights_actual_rapportati.items():
        col1.markdown(f"{chiave}: {round(valore * 100, 2)}%")

    col2.markdown("Pesi Max Sharpe:")
    for chiave, valore in weights_max_sharpe.items():
        col2.markdown(f"{chiave}: {round(valore * 100, 2)}%")

    # Quantità da investire:
    st.markdown("### Numero di azioni da investire secondo l'indice Sharpe Ratio:") 
    alloca = allocazione(matrice_prezzi, weights_max_sharpe, total_portfolio_value)

    styled_df = alloca.style.set_properties(**{
        "background-color": "white", 
        "color": "black", 
        "border-color": "black", 
        'text-align': 'center',
        'width': '250px'
    }).set_table_attributes('style="text-align:center; border-collapse: collapse;"')

    st.write(styled_df.to_html(), unsafe_allow_html=True)