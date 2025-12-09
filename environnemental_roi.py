import streamlit as st
import pandas as pd
import math

# --- 1. LE MOTEUR BACKEND (Ta logique) ---
class GreenROICalculator:
    def __init__(self):
        # Facteurs d'intensité carbone (gCO2e / kWh)
        self.GRID_FACTORS = {
            "France (Nucléaire)": 0.05,
            "USA (Mixte)": 0.38,
            "Chine (Charbon)": 0.53,
            "Inde (Charbon Intensif)": 0.70,
            "Norvège (Hydro)": 0.02,
            "Global (Moyenne)": 0.475
        }
        self.AVG_SALARY = 60000 

    def get_depreciated_value(self, original_price, age_years):
        return original_price * (0.75 ** age_years)

    def calculate_carbon_impact(self, manufacturing_co2, watts, hours_per_day, years_kept, location_name):
        grid_factor = self.GRID_FACTORS.get(location_name, 0.475)
        # (Watts * Heures/Jour * 220 jours * Années) / 1000 => kWh
        total_kwh = (watts * hours_per_day * 220 * years_kept) / 1000
        usage_co2 = total_kwh * grid_factor
        
        return usage_co2, manufacturing_co2 + usage_co2

    def analyze_device(self, device, new_device_benchmark):
        # 1. Productivité (People)
        perf_ratio = device['cpu_score'] / new_device_benchmark
        productivity_loss = 0
        if perf_ratio < 0.5:
            productivity_loss = 0.03 * self.AVG_SALARY
        elif perf_ratio < 0.7:
            productivity_loss = 0.01 * self.AVG_SALARY

        # 2. Finances (Profit)
        resale_value = self.get_depreciated_value(device['original_price'], device['age'])
        net_cost_to_switch = 1500 - resale_value

        # 3. Verdict
        result = {}
        if productivity_loss > net_cost_to_switch:
            result = {"color": "red", "action": "REMPLACER IMMÉDIATEMENT", "msg": f"Perte de productivité ({productivity_loss}€) > Coût matériel."}
        elif "Charbon" in device['location'] and device['watts'] > 150:
             result = {"color": "orange", "action": "PRIORITÉ UPGRADE (ÉNERGIE)", "msg": "Réseau très carboné : l'efficacité énergétique est prioritaire."}
        elif device['age'] < 4:
            result = {"color": "green", "action": "GARDER / UPGRADE RAM", "msg": "L'impact fabrication (300kg CO2) est trop élevé pour changer maintenant."}
        else:
            result = {"color": "yellow", "action": "PLANIFIER REMPLACEMENT", "msg": "Obsolescence proche."}
            
        return result, productivity_loss, net_cost_to_switch, resale_value

# --- 2. L'INTERFACE STREAMLIT ---
st.set_page_config(page_title="Green IT ROI Calculator", layout="wide")
st.title("🌍 Green IT ROI Calculator")
st.markdown("Analysez vos équipements selon les 3 piliers : **People, Planet, Profit**.")

# Initialisation du moteur
engine = GreenROICalculator()

# --- SIDEBAR : PARAMÈTRES GLOBAUX ---
with st.sidebar:
    st.header("⚙️ Configuration")
    location = st.selectbox("Localisation du Bureau", list(engine.GRID_FACTORS.keys()))
    new_cpu_score = st.number_input("Score PassMark Standard (Neuf)", value=25000, step=1000)
    st.info(f"Intensité Réseau : {engine.GRID_FACTORS[location]} kgCO2e/kWh")

# --- SECTION PRINCIPALE : SIMULATEUR ---
st.subheader("💻 Simulateur d'Appareil")

col1, col2, col3, col4 = st.columns(4)
with col1:
    dev_name = st.text_input("Nom du Modèle", "Vieux Laptop Graphiste")
    dev_watts = st.number_input("Conso (Watts)", value=180)
with col2:
    dev_price = st.number_input("Prix d'achat (€)", value=2000)
    dev_hours = st.slider("Heures / Jour", 1, 24, 8)
with col3:
    dev_age = st.number_input("Âge (Années)", value=5)
with col4:
    dev_score = st.number_input("Score PassMark Actuel", value=6900)

# Bouton d'action
if st.button("Calculer le ROI & l'Impact"):
    
    # Création de l'objet device
    device_data = {
        "original_price": dev_price,
        "age": dev_age,
        "cpu_score": dev_score,
        "watts": dev_watts,
        "hours_per_day": dev_hours,
        "location": location
    }

    # Calculs
    verdict, prod_loss, switch_cost, resale = engine.analyze_device(device_data, new_cpu_score)
    
    # Calculs Carbone Spécifiques pour affichage
    co2_usage_1an, co2_total_old = engine.calculate_carbon_impact(0, dev_watts, dev_hours, 1, location)
    # Comparaison avec un neuf (hypothese: 300kg fab + 20% moins de conso)
    co2_usage_new_1an, co2_total_new = engine.calculate_carbon_impact(300, dev_watts*0.8, dev_hours, 1, location)

    # --- AFFICHAGE DES RÉSULTATS ---
    st.divider()
    
    # 1. Le Verdict Visuel
    if verdict["color"] == "red":
        st.error(f"### {verdict['action']}\n{verdict['msg']}")
    elif verdict["color"] == "orange":
        st.warning(f"### {verdict['action']}\n{verdict['msg']}")
    elif verdict["color"] == "green":
        st.success(f"### {verdict['action']}\n{verdict['msg']}")
    else:
        st.warning(f"### {verdict['action']}\n{verdict['msg']}")

    # 2. Les Métriques Clés (3 Piliers)
    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Profit (Coût Net Changement)", f"{switch_cost:.0f} €", f"Revente estimée: {resale:.0f} €")
    c2.metric("👥 People (Perte Productivité)", f"{prod_loss:.0f} € / an", delta_color="inverse")
    c3.metric("🌍 Planet (Dette Carbone 1 an)", f"{co2_usage_1an:.1f} kgCO2e", help="Électricité uniquement")

    # 3. Comparatif Graphique (Old vs New)
    st.subheader("VS : Garder vs Acheter Neuf (Impact sur 1 an)")
    
    chart_data = pd.DataFrame({
        "Scénario": ["Garder l'ancien (Usage pur)", "Acheter Neuf (Fab + Usage)"],
        "Impact CO2 (kg)": [co2_usage_1an, co2_total_new]
    })
    
    st.bar_chart(chart_data, x="Scénario", y="Impact CO2 (kg)", color="#00CC96")
    
    if co2_usage_1an > co2_total_new:
        st.error(f"⚠️ ATTENTION : À cause du réseau électrique ({location}), votre vieux PC pollue plus en 1 an d'électricité que la fabrication complète d'un PC neuf !")
