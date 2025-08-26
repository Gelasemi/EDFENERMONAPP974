import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CHARGEMENT DU MODÈLE ET DES DONNÉES
# =============================================================================

# Charger le modèle avec cache pour éviter de recharger à chaque interaction
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("lstm_model_r2_096.h5")
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# Charger le scaler et les métadonnées
@st.cache_resource
def load_scaler_and_metadata():
    try:
        scaler = joblib.load("scaler_lstm.pkl")
        metadata = joblib.load("lstm_metadata.pkl")
        return scaler, metadata
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None, None

# Charger tous les éléments
model = load_model()
scaler, metadata = load_scaler_and_metadata()

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def generate_historical_data(prediction_date, time_steps, user_params):
    """
    Génère les données historiques des 14 jours précédents en fonction des paramètres utilisateur
    """
    historical_dates = pd.date_range(
        start=prediction_date - timedelta(days=time_steps),
        end=prediction_date - timedelta(days=1),
        freq='D'
    )
    
    # Générer des données basées sur les paramètres utilisateur
    data = {}
    
    # Variables temporelles
    data['Année'] = historical_dates.year
    data['year'] = historical_dates.year
    
    # Variables de production (basées sur les paramètres utilisateur)
    data['Energie Fossile'] = np.random.normal(
        user_params['energie_fossile'], 
        user_params['energie_fossile'] * 0.1, 
        time_steps
    )
    data['Photo Voltaique'] = np.random.normal(
        user_params['photo_voltaique'], 
        user_params['photo_voltaique'] * 0.2, 
        time_steps
    )
    data['Eolien'] = np.random.normal(
        user_params['eolien'], 
        user_params['eolien'] * 0.3, 
        time_steps
    )
    data['Production MGWH'] = np.random.normal(
        user_params['production_mgwh'], 
        user_params['production_mgwh'] * 0.1, 
        time_steps
    )
    
    # Variables de consommation
    data['RES'] = np.random.normal(
        user_params['res'], 
        user_params['res'] * 0.1, 
        time_steps
    )
    data['ENT_PRO'] = np.random.normal(
        user_params['ent_pro'], 
        user_params['ent_pro'] * 0.1, 
        time_steps
    )
    data['ENT'] = np.random.normal(
        user_params['ent'], 
        user_params['ent'] * 0.1, 
        time_steps
    )
    
    # Variables démographiques
    data['Population'] = user_params['population']
    
    # Variables retardées (générées avec une tendance)
    base_consumption = user_params['base_consumption']
    trend = np.linspace(0, user_params['consumption_trend'], time_steps)
    data['lag_1'] = base_consumption + trend + np.random.normal(0, 50000, time_steps)
    data['lag_7'] = base_consumption + trend * 0.9 + np.random.normal(0, 60000, time_steps)
    data['lag_30'] = base_consumption + trend * 0.8 + np.random.normal(0, 70000, time_steps)
    
    # Variables de lissage
    data['rolling_mean_7'] = data['lag_1'].rolling(window=min(7, time_steps)).mean().fillna(method='bfill')
    data['rolling_mean_30'] = data['lag_1'].rolling(window=min(30, time_steps)).mean().fillna(method='bfill')
    data['ema_7'] = data['lag_1'].ewm(span=min(7, time_steps)).mean().fillna(method='bfill')
    
    # Variables d'interaction
    data['event_impact'] = np.random.uniform(0, user_params['event_impact'], time_steps)
    data['res_ratio'] = data['RES'] / (data['Production MGWH'] + 1e-6)
    data['ent_pro_ratio'] = data['ENT_PRO'] / (data['Production MGWH'] + 1e-6)
    data['ent_ratio'] = data['ENT'] / (data['Production MGWH'] + 1e-6)
    
    return pd.DataFrame(data)

def prepare_data_for_prediction(input_data, scaler, metadata):
    """
    Prépare les données pour la prédiction
    """
    # Sélectionner les features dans le bon ordre
    features = metadata['selected_features']
    X_input = input_data[features]
    
    # Normaliser les données
    X_input_scaled = scaler.transform(X_input)
    
    # Reshape pour le LSTM (1 séquence de 14 jours avec 20 features)
    time_steps = metadata['time_steps']
    X_input_reshaped = X_input_scaled.reshape(1, time_steps, len(features))
    
    return X_input_reshaped

# =============================================================================
# INTERFACE UTILISATEUR
# =============================================================================

# Configuration de la page
st.set_page_config(
    page_title="Prédiction LSTM R² = 0.96",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .param-section {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown("<h1 class='main-header'>⚡ Prédiction Énergétique avec LSTM</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Modèle performant (R² = 0.96) pour la prévision de consommation</p>", unsafe_allow_html=True)

# Vérifier que le modèle est chargé
if model is None or scaler is None or metadata is None:
    st.error("""
    ### ❌ Fichiers du modèle non trouvés
    
    Veuillez vous assurer que les fichiers suivants sont présents:
    - `lstm_model_r2_096.h5`
    - `scaler_lstm.pkl`
    - `lstm_metadata.pkl`
    """)
    st.stop()

# Afficher les informations du modèle
st.sidebar.success(f"✅ Modèle LSTM chargé (R² = {metadata['model_metrics']['r2']:.4f})")

# =============================================================================
# SECTION DE PRÉDICTION
# =============================================================================

st.header("🎯 Interface de Prédiction")

# Date de prédiction
prediction_date = st.date_input(
    "📅 Sélectionnez la date de prédiction",
    min_value=datetime.now().date() + timedelta(days=1),
    max_value=datetime.now().date() + timedelta(days=90),
    value=datetime.now().date() + timedelta(days=7)
)

# Paramètres utilisateur
st.subheader("📊 Paramètres de prédiction")

# Créer des colonnes pour organiser les paramètres
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='param-section'>", unsafe_allow_html=True)
    st.markdown("#### 🌡️ Paramètres Météo")
    temperature = st.slider("Température moyenne (°C)", 15.0, 35.0, 25.0, step=0.1)
    humidity = st.slider("Humidité moyenne (%)", 30, 90, 65)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='param-section'>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Paramètres Énergétiques")
    energie_fossile = st.number_input("Énergie Fossile (MW)", 1000, 3000, 1800)
    photo_voltaique = st.number_input("Photo Voltaique (MW)", 0, 500, 150)
    eolien = st.number_input("Éolien (MW)", 0, 50, 10)
    production_mgwh = st.number_input("Production Totale (MGWh)", 1500, 3500, 2200)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='param-section'>", unsafe_allow_html=True)
    st.markdown("#### 👥 Paramètres Socio-économiques")
    population = st.number_input("Population", 700000, 1000000, 850000)
    base_consumption = st.number_input("Consommation de base (kWh)", 1500000, 3000000, 2000000)
    consumption_trend = st.number_input("Tendance de consommation", -100000, 100000, 0)
    event_impact = st.number_input("Impact des événements", 0, 100000, 25000)
    st.markdown("</div>", unsafe_allow_html=True)

# Paramètres de consommation détaillés
st.subheader("📈 Paramètres de Consommation Détaillés")

col4, col5, col6 = st.columns(3)

with col4:
    res = st.number_input("Consommation RES (kWh)", 800000, 1500000, 1100000)
    ent_pro = st.number_input("Consommation ENT_PRO (kWh)", 200000, 600000, 350000)

with col5:
    ent = st.number_input("Consommation ENT (kWh)", 800000, 1500000, 1100000)

with col6:
    st.markdown("<div style='padding-top: 20px;'>", unsafe_allow_html=True)
    st.info("💡 Ces paramètres seront utilisés pour générer les 14 jours de données historiques nécessaires à la prédiction.")
    st.markdown("</div>", unsafe_allow_html=True)

# Bouton de prédiction
if st.button("🚀 Lancer la Prédiction", type="primary", use_container_width=True):
    with st.spinner("🔄 Préparation des données et calcul de la prédiction..."):
        # Créer le dictionnaire des paramètres utilisateur
        user_params = {
            'temperature': temperature,
            'humidity': humidity,
            'energie_fossile': energie_fossile,
            'photo_voltaique': photo_voltaique,
            'eolien': eolien,
            'production_mgwh': production_mgwh,
            'population': population,
            'base_consumption': base_consumption,
            'consumption_trend': consumption_trend,
            'event_impact': event_impact,
            'res': res,
            'ent_pro': ent_pro,
            'ent': ent
        }
        
        # Générer les données historiques
        time_steps = metadata['time_steps']
        historical_data = generate_historical_data(prediction_date, time_steps, user_params)
        
        # Afficher un aperçu des données générées
        st.subheader("📋 Aperçu des données générées")
        st.write(f"Données historiques des {time_steps} jours précédant le {prediction_date.strftime('%d/%m/%Y')}:")
        st.dataframe(historical_data)
        
        # Préparer les données pour la prédiction
        X_input_reshaped = prepare_data_for_prediction(historical_data, scaler, metadata)
        
        # Faire la prédiction
        prediction = model.predict(X_input_reshaped)[0][0]
        
        # Afficher les résultats
        st.success("### 🎯 Résultat de la Prédiction")
        
        # Afficher les métriques dans des cartes
        col_result1, col_result2, col_result3 = st.columns(3)
        
        with col_result1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("📅 Date", prediction_date.strftime('%d/%m/%Y'))
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_result2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("⚡ Consommation prédite", f"{prediction:,.0f} kWh")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_result3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric("📊 Précision (R²)", f"{metadata['model_metrics']['r2']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Visualisation
        st.subheader("📈 Visualisation de la Prédiction")
        
        # Créer un graphique avec les données historiques et la prédiction
        fig = go.Figure()
        
        # Ajouter les données historiques générées
        historical_dates = pd.date_range(
            start=prediction_date - timedelta(days=time_steps),
            end=prediction_date - timedelta(days=1),
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_data['lag_1'],
            mode='lines+markers',
            name='Données historiques',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Ajouter la prédiction
        fig.add_trace(go.Scatter(
            x=[prediction_date],
            y=[prediction],
            mode='markers',
            name='Prédiction LSTM',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title=f"Prédiction de consommation pour le {prediction_date.strftime('%d/%m/%Y')}",
            xaxis_title="Date",
            yaxis_title="Consommation (kWh)",
            hovermode='x unified',
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Informations supplémentaires
        st.subheader("ℹ️ Informations sur la prédiction")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            **Paramètres utilisés pour la génération des données:**
            - Température moyenne: {}°C
            - Population: {:,} habitants
            - Production totale: {:,} MGWh
            - Impact des événements: {:,} kWh
            """.format(
                temperature, population, production_mgwh, event_impact
            ))
        
        with col_info2:
            st.markdown("""
            **Performance du modèle:**
            - R²: {:.4f}
            - MAE: {:,.0f} kWh
            - RMSE: {:,.0f} kWh
            - Architecture: {} unités LSTM
            """.format(
                metadata['model_metrics']['r2'],
                metadata['model_metrics']['mae'],
                metadata['model_metrics']['rmse'],
                metadata['model_architecture']['lstm_units']
            ))

# =============================================================================
# SECTION D'INFORMATION
# =============================================================================

st.header("📚 À propos du modèle")

with st.expander("🧠 Architecture du modèle LSTM", expanded=True):
    st.markdown("""
    ### Caractéristiques techniques du modèle:
    
    **Architecture:**
    - Première couche LSTM: 64 unités avec return_sequences=True
    - Dropout: 30% après chaque couche LSTM
    - Deuxième couche LSTM: 32 unités
    - Couche de sortie: 1 neurone (régression)
    
    **Entraînement:**
    - Optimiseur: Adam
    - Fonction de perte: Mean Squared Error (MSE)
    - Callbacks: EarlyStopping (patience=3)
    - Validation split: 10%
    
    **Performance:**
    - R²: {:.4f}
    - MAE: {:,.0f} kWh
    - RMSE: {:,.0f} kWh
    """.format(
        metadata['model_metrics']['r2'],
        metadata['model_metrics']['mae'],
        metadata['model_metrics']['rmse']
    ))

with st.expander("📊 Features utilisées (20 variables)"):
    st.markdown("""
    Le modèle utilise les 20 features suivantes:
    
    **Variables temporelles:**
    - Année, year
    
    **Variables de production:**
    - Énergie Fossile, Photo Voltaique, Eolien, Production MGWH
    
    **Variables de consommation:**
    - RES, ENT_PRO, ENT
    
    **Variables démographiques:**
    - Population
    
    **Variables retardées:**
    - lag_1, lag_7, lag_30
    
    **Variables de lissage:**
    - rolling_mean_7, rolling_mean_30, ema_7
    
    **Variables d'interaction:**
    - event_impact, res_ratio, ent_pro_ratio, ent_ratio
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 Application de prédiction énergétique avec modèle LSTM</p>
    <p>📊 R² = {:.4f} | MAE = {:,.0f} kWh</p>
    <p>Développé pour EnerMonito</p>
</div>
""".format(
    metadata['model_metrics']['r2'],
    metadata['model_metrics']['mae']
), unsafe_allow_html=True)