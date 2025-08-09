import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# App setup
# =========================
st.set_page_config(page_title="Plant Growth — Pro, Clear & Visual", layout="wide")
st.title("🌱 Plant Growth Simulator — Pro, Clear & Visual")

# Deterministic unless user changes seed
DEFAULT_SEED = 42

# =========================
# Helper: simulate one plant
# =========================
def simulate_plant(
    days:int,
    seed:int,
    plant_name:str,
    growth_speed:float,     # how fast it grows when conditions are good
    thirst_factor:float,    # how quickly it uses water (1.0 = normal, >1 = thirstier)
    fertilizer_boost:float, # multiplier on growth from fertilizer (1.2 = +20%)
    use_fertilizer:bool,
    soil_capacity:float,    # max liters of water soil can hold per m^2
    start_soil:float,       # starting liters in soil
    # weather
    avg_sun:float, sun_var:float,
    avg_temp:float, temp_var:float,
    rain_prob:float, rain_min:float, rain_max:float, clustered:bool,
    # watering
    mode:str,               # "auto" or "manual"
    threshold:float,        # auto: water when soil < threshold
    auto_amount:float,      # auto: liters to add
    manual_days:list,       # manual watering days (1..days)
    manual_amount:float,    # liters per manual watering
    # events
    pest_prob:float, pest_penalty:float
):
    rng = np.random.default_rng(seed)
    days_list = np.arange(1, days+1)

    # Seasonal-ish wiggle across the window to avoid boring flat weather
    drift = np.sin(np.linspace(0, 2*np.pi, days))  # -1..1

    sun = np.clip(rng.normal(avg_sun + drift*0.6, sun_var, size=days), 0, 14)
    temp = np.clip(rng.normal(avg_temp + drift*1.2, temp_var, size=days), 0, 45)

    # Rain with optional clustering (rain is more likely after rain)
    rain = np.zeros(days)
    was_rainy = False
    for i in range(days):
        p = rain_prob + (0.15 if (clustered and was_rainy) else -0.1 if clustered else 0.0)
        p = np.clip(p, 0, 1)
        drop = rng.random() < p
        if drop:
            rain[i] = rng.uniform(rain_min, rain_max)
        was_rainy = drop

    # Initialize
    soil = min(max(start_soil, 0.0), soil_capacity)
    height = 0.0
    fert_mult = fertilizer_boost if use_fertilizer else 1.0

    # Storage
    records = []

    for d in range(days):
        today = d+1
        s = float(sun[d])
        t = float(temp[d])
        r = float(rain[d])

        # Watering
        water_add = 0.0
        if mode == "auto":
            if soil < threshold:
                water_add = auto_amount
        else:
            if today in manual_days:
                water_add = manual_amount

        # Update soil (rain + watering)
        soil += r + water_add
        # Cap (excess runs off)
        if soil > soil_capacity:
            soil = soil_capacity

        # Daily water loss:
        # - Sunny days and higher temperatures evaporate more water
        # - thirst_factor makes some plants (e.g., herbs) use water faster
        water_loss = (s * 0.35 + max(0.0, t - 10.0) * 0.05) * thirst_factor
        soil = max(0.0, soil - water_loss)

        # Growth:
        # - Needs some minimum soil water; more sun helps up to a point
        # - Fertilizer gives a gentle boost
        # - Random pest hit slightly reduces growth that day
        pest_hit = (rng.random() < pest_prob)
        pest_mult = (pest_penalty if pest_hit else 1.0)

        if soil < 5.0:
            daily_growth = 0.0  # too dry to grow
        else:
            # Growth increases with sun until ~10 hours, then plateaus
            sun_factor = min(1.0, s / 10.0)
            daily_growth = growth_speed * sun_factor * fert_mult * pest_mult

        height += daily_growth

        records.append({
            "Day": today,
            "Plant": plant_name,
            "Sunlight (h)": round(s, 2),
            "Temperature (°C)": round(t, 1),
            "Rain (L/m²)": round(r, 2),
            "Watering (L/m²)": round(water_add, 2),
            "Soil Water (L/m²)": round(soil, 2),
            "Daily Growth (cm)": round(daily_growth, 2),
            "Total Height (cm)": round(height, 2),
            "Pest?": int(pest_hit)
        })

    df = pd.DataFrame(records)
    return df

# =========================
# Sidebar — Global settings
# =========================
st.sidebar.header("Global Simulation Settings")

DAYS = st.sidebar.slider("Days to Simulate", 20, 120, 45, 5)
seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=999999, value=DEFAULT_SEED, step=1)

st.sidebar.markdown("### Weather")
avg_sun = st.sidebar.slider("Average Sunlight (hours/day)", 4.0, 12.0, 8.0, 0.5)
sun_var = st.sidebar.slider("Sunlight Variation (± hours)", 0.0, 3.0, 1.2, 0.1)

avg_temp = st.sidebar.slider("Average Temperature (°C)", 10.0, 35.0, 25.0, 0.5)
temp_var = st.sidebar.slider("Temperature Variation (± °C)", 0.0, 6.0, 2.0, 0.5)

st.sidebar.markdown("### Rain")
rain_prob = st.sidebar.slider("Chance of Rain (%)", 0, 100, 40, 1) / 100.0
rain_min, rain_max = st.sidebar.slider("Rain Amount Range (L/m²)", 0, 40, (3, 12), 1)
clustered = st.sidebar.checkbox("Clustered Storms (rainy days often follow rainy days)", True)

st.sidebar.markdown("### Soil Water (simple bucket)")
soil_capacity = st.sidebar.slider("Soil Water Capacity (L/m²)", 10, 60, 30, 1,
                                  help="How much water the soil can hold in total.")
start_soil = st.sidebar.slider("Starting Water in Soil (L/m²)", 0, 60, 18, 1)

st.sidebar.markdown("### Events")
pest_prob = st.sidebar.slider("Daily Pest Chance", 0.0, 0.3, 0.05, 0.01)
pest_penalty = st.sidebar.slider("Pest Effect on Growth (multiplier)", 0.3, 1.0, 0.75, 0.05)

# =========================
# Plant presets (plain English)
# =========================
PLANT_PRESETS = {
    "Flower": {
        "growth_speed": 1.6,   # cm/day potential when conditions are good
        "thirst_factor": 1.0,  # normal water use
        "fert_boost": 1.20
    },
    "Tree": {
        "growth_speed": 1.1,
        "thirst_factor": 1.2,  # uses a bit more water
        "fert_boost": 1.15
    },
    "Herb": {
        "growth_speed": 1.9,
        "thirst_factor": 1.3,  # fast grower, thirstier
        "fert_boost": 1.25
    },
    "Cactus": {
        "growth_speed": 0.8,
        "thirst_factor": 0.5,  # sips water; slow grower
        "fert_boost": 1.10
    }
}

# =========================
# Sidebar — Plants (up to 3 scenarios)
# =========================
st.sidebar.markdown("---")
st.sidebar.header("Plant Scenarios (up to 3)")

def plant_block(label_default, default_type="Flower", default_mode="auto"):
    st.sidebar.markdown(f"{label_default}")
    ptype = st.sidebar.selectbox(f"Plant Type — {label_default}",
                                 list(PLANT_PRESETS.keys()), index=list(PLANT_PRESETS.keys()).index(default_type))
    use_fert = st.sidebar.checkbox(f"Fertilizer on Day 1 — {label_default}", True, key=f"fert_{label_default}")
    mode = st.sidebar.radio(f"Watering Mode — {label_default}", ["auto", "manual"], index=(0 if default_mode=="auto" else 1), key=f"mode_{label_default}")
    if mode == "auto":
        threshold = st.sidebar.slider(f"Water when soil below (L/m²) — {label_default}", 2, 30, 12, 1, key=f"thr_{label_default}")
        amount = st.sidebar.slider(f"Auto watering amount (L/m²) — {label_default}", 2, 30, 10, 1, key=f"amt_{label_default}")
        manual_days = []
        manual_amount = 0
    else:
        threshold = 0
        amount = 0
        manual_days = st.sidebar.multiselect(f"Manual watering days (1..{DAYS}) — {label_default}",
                                             list(range(1, DAYS+1)), default=[7, 14, 21, 28], key=f"days_{label_default}")
        manual_amount = st.sidebar.slider(f"Manual watering amount (L/m²) — {label_default}", 2, 30, 12, 1, key=f"mamt_{label_default}")
    st.sidebar.markdown("---")
    return {
        "label": label_default,
        "ptype": ptype,
        "use_fert": use_fert,
        "mode": mode,
        "threshold": threshold,
        "amount": amount,
        "manual_days": manual_days,
        "manual_amount": manual_amount
    }

plantA = plant_block("Plant A", default_type="Flower", default_mode="auto")
plantB = plant_block("Plant B", default_type="Herb", default_mode="auto")
use_third = st.sidebar.checkbox("Add Plant C", False)
plantC = None
if use_third:
    plantC = plant_block("Plant C", default_type="Tree", default_mode="manual")

# =========================
# Simulate all selected plants
# =========================
def run_one(label, cfg):
    preset = PLANT_PRESETS[cfg["ptype"]]
    df = simulate_plant(
        days=DAYS,
        seed=seed + hash(label)%1000,   # nudge seed per plant for variety
        plant_name=f"{label} ({cfg['ptype']})",
        growth_speed=preset["growth_speed"],
        thirst_factor=preset["thirst_factor"],
        fertilizer_boost=preset["fert_boost"],
        use_fertilizer=cfg["use_fert"],
        soil_capacity=soil_capacity,
        start_soil=start_soil,
        avg_sun=avg_sun, sun_var=sun_var,
        avg_temp=avg_temp, temp_var=temp_var,
        rain_prob=rain_prob, rain_min=rain_min, rain_max=rain_max, clustered=clustered,
        mode=cfg["mode"],
        threshold=cfg["threshold"],
        auto_amount=cfg["amount"],
        manual_days=cfg["manual_days"],
        manual_amount=cfg["manual_amount"],
        pest_prob=pest_prob, pest_penalty=pest_penalty
    )
    return df

dfs = []
dfs.append(run_one(plantA["label"], plantA))
dfs.append(run_one(plantB["label"], plantB))
if plantC:
    dfs.append(run_one(plantC["label"], plantC))

df_all = pd.concat(dfs, ignore_index=True)

# =========================
# KPIs per plant
# =========================
kcols = st.columns(3 if plantC else 2)
plants_list = df_all["Plant"].unique().tolist()

for i, plant in enumerate(plants_list):
    d = df_all[df_all["Plant"] == plant]
    final_h = d["Total Height (cm)"].iloc[-1]
    total_rain = d["Rain (L/m²)"].sum()
    total_water = d["Watering (L/m²)"].sum()
    # very rough "water productivity": height per (rain + watering)
    water_used = total_rain + total_water if (total_rain + total_water) > 0 else 1.0
    wp = final_h / water_used
    kcols[i].metric(f"{plant} — Final Height", f"{final_h:.1f} cm")
    kcols[i].metric(f"{plant} — Total Rain", f"{total_rain:.1f} L/m²")
    kcols[i].metric(f"{plant} — Total Watering", f"{total_water:.1f} L/m²")
    kcols[i].metric(f"{plant} — Height per Water", f"{wp:.3f} cm per L")

st.caption("Notes: This model is intentionally simple and readable: rain + watering fill the soil bucket, sun/heat dry it, "
           "and growth depends on having enough soil water plus sunlight. Fertilizer and pests gently push results up or down.")

# =========================
# TABS: Overview | Water & Inputs | Weather | Growth Analysis | Data
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Water & Inputs", "Weather", "Growth Analysis", "Data"])

# ---------- TAB 1: Overview (Heights & Soil) ----------
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📈 Plant Height Over Time")
        fig, ax = plt.subplots()
        for plant in plants_list:
            d = df_all[df_all["Plant"] == plant]
            ax.plot(d["Day"], d["Total Height (cm)"], marker="o", label=plant)
        ax.set_xlabel("Day"); ax.set_ylabel("Height (cm)")
        ax.grid(True); ax.legend()
        st.pyplot(fig)

    with c2:
        st.subheader("💧 Soil Water Over Time")
        fig2, ax2 = plt.subplots()
        for plant in plants_list:
            d = df_all[df_all["Plant"] == plant]
            ax2.plot(d["Day"], d["Soil Water (L/m²)"], marker="o", label=plant)
        ax2.set_xlabel("Day"); ax2.set_ylabel("Soil Water (L/m²)")
        ax2.grid(True); ax2.legend()
        st.pyplot(fig2)

# ---------- TAB 2: Water & Inputs ----------
with tab2:
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("🌧 Rain vs. Watering (per day)")
        # Rain is identical across plants because it's global weather.
        # Watering is plant-specific (auto/manual).
        days_unique = sorted(df_all["Day"].unique())
        rain_avg = df_all.groupby("Day")["Rain (L/m²)"].mean()
        fig3, ax3 = plt.subplots()
        ax3.bar(days_unique, rain_avg.loc[days_unique], label="Rain")
        # Plot watering as lines per plant for clarity
        for plant in plants_list:
            d = df_all[df_all["Plant"] == plant]
            ax3.plot(d["Day"], d["Watering (L/m²)"], marker="o", label=f"Watering — {plant}")
        ax3.set_xlabel("Day"); ax3.set_ylabel("L/m²")
        ax3.grid(True); ax3.legend()
        st.pyplot(fig3)

    with c4:
        st.subheader("🚿 Cumulative Water & Height")
        fig4, ax4 = plt.subplots()
        for plant in plants_list:
            d = df_all[df_all["Plant"] == plant].copy()
            d["Cumulative Water (L/m²)"] = (d["Rain (L/m²)"] + d["Watering (L/m²)"]).cumsum()
            ax4.plot(d["Cumulative Water (L/m²)"], d["Total Height (cm)"], marker="o", label=plant)
        ax4.set_xlabel("Cumulative Water (L/m²)"); ax4.set_ylabel("Height (cm)")
        ax4.grid(True); ax4.legend()
        st.pyplot(fig4)

    st.subheader("📦 Water Inputs Table (Totals)")
    summary = df_all.groupby("Plant")[["Rain (L/m²)", "Watering (L/m²)"]].sum()
    summary["Total Water (L/m²)"] = summary["Rain (L/m²)"] + summary["Watering (L/m²)"]
    st.dataframe(summary.round(2), use_container_width=True)

# ---------- TAB 3: Weather ----------
with tab3:
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("☀ Sunlight by Day")
        # Same for all plants; show average per day
        sun_avg = df_all.groupby("Day")["Sunlight (h)"].mean()
        fig5, ax5 = plt.subplots()
        ax5.plot(sun_avg.index, sun_avg.values, marker="o")
        ax5.set_xlabel("Day"); ax5.set_ylabel("Hours")
        ax5.grid(True)
        st.pyplot(fig5)

    with c6:
        st.subheader("🌡 Temperature by Day")
        temp_avg = df_all.groupby("Day")["Temperature (°C)"].mean()
        fig6, ax6 = plt.subplots()
        ax6.plot(temp_avg.index, temp_avg.values, marker="o")
        ax6.set_xlabel("Day"); ax6.set_ylabel("°C")
        ax6.grid(True)
        st.pyplot(fig6)

    st.subheader("📊 Weather Distributions")
    c7, c8 = st.columns(2)
    with c7:
        fig7, ax7 = plt.subplots()
        ax7.hist(df_all["Sunlight (h)"], bins=12)
        ax7.set_xlabel("Sunlight (h)"); ax7.set_ylabel("Count")
        ax7.set_title("Histogram: Sunlight")
        st.pyplot(fig7)
    with c8:
        fig8, ax8 = plt.subplots()
        ax8.hist(df_all["Temperature (°C)"], bins=12)
        ax8.set_xlabel("Temperature (°C)"); ax8.set_ylabel("Count")
        ax8.set_title("Histogram: Temperature")
        st.pyplot(fig8)

# ---------- TAB 4: Growth Analysis ----------
with tab4:
    c9, c10 = st.columns(2)

    with c9:
        st.subheader("📈 Daily Growth by Plant")
        fig9, ax9 = plt.subplots()
        for plant in plants_list:
            d = df_all[df_all["Plant"] == plant]
            ax9.plot(d["Day"], d["Daily Growth (cm)"], marker="o", label=plant)
        ax9.set_xlabel("Day"); ax9.set_ylabel("Growth (cm)")
        ax9.grid(True); ax9.legend()
        st.pyplot(fig9)

    with c10:
        st.subheader("📉 Soil Water vs. Daily Growth (relationship)")
        fig10, ax10 = plt.subplots()
        for plant in plants_list:
            d = df_all[df_all["Plant"] == plant]
            ax10.scatter(d["Soil Water (L/m²)"], d["Daily Growth (cm)"], label=plant)
        ax10.set_xlabel("Soil Water (L/m²)"); ax10.set_ylabel("Daily Growth (cm)")
        ax10.grid(True); ax10.legend()
        st.pyplot(fig10)

    st.subheader("📦 Growth Stats by Plant")
    stats = df_all.groupby("Plant")[["Daily Growth (cm)", "Total Height (cm)"]].agg(
        DailyGrowth_Avg=("Daily Growth (cm)", "mean"),
        DailyGrowth_Max=("Daily Growth (cm)", "max"),
        FinalHeight=("Total Height (cm)", "last")
    )
    st.dataframe(stats.round(2), use_container_width=True)

# ---------- TAB 5: Data ----------
with tab5:
    st.subheader("Combined Simulation Table")
    st.dataframe(df_all, use_container_width=True)

    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", data=csv, file_name="plant_growth_sim_all.csv", mime="text/csv")