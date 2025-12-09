import streamlit as st
import pandas as pd
import requests # Pour appeler l'API Boavizta
import math

# --- 1. FONCTIONS DE RÉCUPÉRATION DE DONNÉES (API & CSV) ---

@st.cache_data # Streamlit garde le résultat en mémoire pour aller plus vite
def get_boavizta_footprint(model_name):
    """
    Interroge l'API Boavizta pour obtenir l'impact carbone de fabrication (GWP).
    Note: Ceci est une version simplifiée utilisant leur base de données publique.
    """
    # Endpoint de recherche (exemple simplifié)
    url = f"https://api.boavizta.org/v1/component/search?name={model_name}"
    
    try:
        # Dans un vrai cas, on envoie souvent des specs précises (RAM, SSD, etc.)
        # Ici on simule une récupération ou on met une valeur par défaut si l'API échoue
        # Pour l'exercice, si l'API ne répond pas, on renvoie une estimation
        # response = requests.get(url) 
        # data = response.json()
        
        # Simulation de réponse API pour que tu puisses tester sans clé API immédiate
        # Si tu as une clé, décommente les lignes requests au-dessus
        return 320.0 # Valeur retournée par l'API (ex: 320 kg CO2e)
        
    except Exception as e:
        st.error(f"Erreur connexion Boavizta: {e}")
        return 300.0 # Valeur de repli par défaut

@st.cache_data
def get_cpu_score(cpu_name):
    """
    Simule la lecture de ton fichier CSV PassMark.
    Dans la réalité: df = pd.read_csv('passmark_data.csv')
    """
    # Exemple de ta stratégie "Hacker" (Chargement du CSV en mémoire)
    # Ici je crée un petit dictionnaire pour l'exemple, mais tu chargeras ton CSV
    fake_csv_db = {
        "i5-8250U": 6000,
        "i7-7700HQ": 6900,
        "i7-1355U": 15000,
        "M3 Max": 35000,
        "M1": 14000
    }
    return fake_csv_db.get(cpu_name, 5000) # 5000 par défaut si non trouvé

# --- 2. LE MOTEUR DE CALCUL (Mis à jour) ---

class GreenROICalculator:
    def __init__(self):
        self.GRID_FACTORS = {
            "France (Nucléaire)": 0.05,
            "USA (Mixte)": 0.38,
            "Chine (Charbon)": 0.53,
            "Inde (Charbon Intensif)": 0.70,
            "Global (Moyenne)": 0.475
        }
        self.AVG_SALARY = 60000 

    def get_depreciated_value(self, original_price, age_years):
        return original_price * (0.75 ** age_years)

    def calculate_carbon_impact(self, manufacturing_co2, watts, hours_per_day, years_kept, location_name):
        grid_factor = self.GRID_FACTORS.get(location_name, 0.475)
        total_kwh = (watts * hours_per_day * 220 * years_kept) / 1000
        usage_co2 = total_kwh * grid_factor
        return usage_co2, manufacturing_co2 + usage_co2

    def analyze_device(self, device, new_device_benchmark, manufacturing_co2_from_api):
        # 1. Productivité (Utilisation du score CPU réel)
        perf_ratio = device['cpu_score'] / new_device_benchmark
        productivity_loss = 0
        if perf_ratio < 0.5:
            productivity_loss = 0.03 * self.AVG_SALARY
        elif perf_ratio < 0.7:
            productivity_loss = 0.01 * self.AVG_SALARY

        # 2. Finances
        resale_value = self.get_depreciated_value(device['original_price'], device['age'])
        net_cost_to_switch = 1500 - resale_value

        # 3. Planet: On vérifie si l'impact usage dépasse la fabrication d'un neuf
        # Coût carbone fabrication d'un neuf (Hypothèse ou API aussi)
        new_mfg_co2 = 300 
        
        result = {}
        # Logique simplifiée pour l'affichage
        if productivity_loss > net_cost_to_switch:
             result = {"color": "red", "action": "REMPLACER (Productivité)", "msg": f"Perte ({productivity_loss}€) > Coût matériel."}
        elif "Charbon" in device['location'] and device['watts'] > 150:
             result = {"color": "orange", "action": "PRIORITÉ UPGRADE (Énergie)", "msg": "Réseau sale : priorité efficacité."}
        elif device['age'] < 4:
            result = {"color": "green", "action": "GARDER (Carbone)", "msg": f"Économisez {new_mfg_co2}kg de CO2 de fabrication."}
        else:
            result = {"color": "yellow", "action": "PLANIFIER", "msg": "Fin de vie proche."}
            
        return result, productivity_loss, net_cost_to_switch, resale_value

# --- 3. L'INTERFACE STREAMLIT ---
st.set_page_config(page_title="Green IT ROI - Live Data", layout="wide")
st.title("🌍 Green IT ROI (API Connected)")

engine = GreenROICalculator()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    location = st.selectbox("Localisation", list(engine.GRID_FACTORS.keys()))
    # On sélectionne le CPU standard actuel via la "Base CSV"
    current_std_cpu = st.selectbox("Standard du marché (CPU)", ["i7-1355U", "M3 Max"])
    new_cpu_score = get_cpu_score(current_std_cpu)
    st.info(f"Score Standard ({current_std_cpu}): {new_cpu_score}")

# Formulaire
st.subheader("🔎 Recherche Appareil (API Boavizta)")
col1, col2 = st.columns(2)

with col1:
    model_query = st.text_input("Modèle Laptop (ex: Dell Latitude 7490)", "Dell Latitude 7490")
    # Simulation recherche CPU dans le CSV
    cpu_query = st.selectbox("Processeur détecté (Simulé CSV)", ["i5-8250U", "i7-7700HQ", "M1"])
    
with col2:
    dev_price = st.number_input("Prix d'achat original (€)", 1500)
    dev_age = st.number_input("Âge (années)", 4)
    dev_watts = st.number_input("Conso (W)", 65)

if st.button("Lancer l'Analyse Live"):
    
    # 1. APPEL API (Récupération Empreinte Carbone)
    with st.spinner('Connexion à Boavizta API...'):
        mfg_co2 = get_boavizta_footprint(model_query)
        st.toast(f"Données Carbone récupérées: {mfg_co2} kgCO2e", icon="🌱")

    # 2. LOOKUP CSV (Récupération Score CPU)
    current_score = get_cpu_score(cpu_query)

    # 3. CALCUL
    device_data = {
        "original_price": dev_price, "age": dev_age, "cpu_score": current_score,
        "watts": dev_watts, "hours_per_day": 8, "location": location
    }
    
    verdict, prod_loss, switch_cost, resale = engine.analyze_device(device_data, new_cpu_score, mfg_co2)
    
    # --- RÉSULTATS ---
    st.divider()
    
    # Affichage dynamique de la source de donnée
    st.caption(f"Sources: Carbone via Boavizta API ({mfg_co2}kg) | Perf via PassMark CSV ({current_score})")

    if verdict["color"] == "red": st.error(f"### {verdict['action']}\n{verdict['msg']}")
    elif verdict["color"] == "green": st.success(f"### {verdict['action']}\n{verdict['msg']}")
    else: st.warning(f"### {verdict['action']}\n{verdict['msg']}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Impact Fabrication (API)", f"{mfg_co2} kgCO2e")
    c2.metric("Impact Usage (1 an)", f"{engine.calculate_carbon_impact(mfg_co2, dev_watts, 8, 1, location)[0]:.1f} kgCO2e")
    c3.metric("ROI Financier", f"{prod_loss - switch_cost:.0f} €", delta="Positif = Changer est rentable")
