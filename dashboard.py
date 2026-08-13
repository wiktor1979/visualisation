import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Monitor Pompy Ciepła", layout="wide", page_icon="🔥")

st.title("🔥 Panel Monitorowania i Diagnostyki Pompy Ciepła")

DB_FILE = "tuya_telemetry.db"

# --- SŁOWNIK METADANYCH PARAMETRÓW ---
PARAM_INFO = {
    "in_water_temp": {"label": "Powrót CO", "desc": "Temperatura wody powracającej z instalacji grzewczej"},
    "out_water_temp": {"label": "Zasilanie CO", "desc": "Temperatura wody wychodzącej na dom"},
    "tank_temp": {"label": "Woda CWU", "desc": "Temperatura wody w zasobniku ciepłej wody użytkowej"},
    "amb_temp": {"label": "Temp. zewnętrzna", "desc": "Temperatura powietrza na zewnątrz budynku"},
    "disc_temp": {"label": "Tłoczenie sprężarki", "desc": "Temperatura gazu na wylocie/tłoczeniu sprężarki (Discharge)"},
    "back_temp": {"label": "Powrót do sprężarki", "desc": "Temperatura czynnika na powrocie do sprężarki (Suction)"},
    "tidr": {"label": "Temp. ssania", "desc": "Temperatura czujnika ssania / wymiennika chłodniczego"},
    "heat_temp_set": {"label": "Nastawa CO", "desc": "Docelowa zadana temperatura dla trybu ogrzewania CO"},
    "cool_temp_set": {"label": "Nastawa Chłodzenia", "desc": "Docelowa zadana temperatura dla trybu chłodzenia"},
    "hot_water_temp_set": {"label": "Nastawa CWU", "desc": "Docelowa zadana temperatura dla wody użytkowej"},
    "ac_vol": {"label": "Napięcie AC", "desc": "Napięcie zasilania sieciowego AC podawane do jednostki"},
    "ac_curr": {"label": "Prąd AC", "desc": "Natężenie prądu pobieranego przez urządzenie"},
    "comp_freq": {"label": "Częstotliwość sprężarki", "desc": "Aktualna częstotliwość pracy sprężarki (Hz)"},
    "flow_rate": {"label": "Przepływ", "desc": "Przepływ wody w obiegu hydraulicznym"},
    "m_eev": {"label": "Zawór EEV główny", "desc": "Pozycja otwarcia głównego elektronicznego zaworu rozprężnego"},
    "valve": {"label": "Zawór 3-drożny", "desc": "Stan zaworu przełączającego (0 = CO, 1 = CWU)"},
    "defrost": {"label": "Odszranianie", "desc": "Cykl automatycznego odszraniania parownika"}
}

def get_param_label(code: str) -> str:
    info = PARAM_INFO.get(code)
    return f"{info['label']} ({code})" if info else code

# --- PANEL BOCZNY ---
st.sidebar.header("⏱️ Zakres danych")
time_range_map = {
    "Ostatnie 6 godzin": 6,
    "Ostatnie 24 godziny": 24,
    "Ostatnie 3 dni": 72,
    "Ostatnie 7 dni": 168
}
selected_range = st.sidebar.selectbox("Wybierz zakres czasu:", list(time_range_map.keys()), index=1)
hours_back = time_range_map[selected_range]

st.sidebar.header("📊 Optymalizacja wykresów")
resample_map = {
    "Brak (Surowe dane)": None,
    "Co 1 minuta": "1min",
    "Co 5 minut": "5min",
    "Co 15 minut": "15min"
}
selected_resample = st.sidebar.selectbox("Agregacja punktów:", list(resample_map.keys()), index=1)
resample_rule = resample_map[selected_resample]

st.sidebar.header("⚙️ Kalkulator COP")
cos_phi = st.sidebar.slider("Współczynnik mocy (cos φ)", 0.80, 1.00, 0.92, 0.01)
ac_curr_div = st.sidebar.selectbox("Dzielnik prądu (ac_curr)", [1, 10, 100], index=1)

def load_data(hours: int) -> pd.DataFrame:
    conn = sqlite3.connect(DB_FILE)
    query = f"""
        SELECT 
            datetime(timestamp, 'unixepoch', 'localtime') as czas,
            code, val_num, val_str
        FROM telemetry
        WHERE timestamp >= strftime('%s', 'now', '-{hours} hours')
        ORDER BY timestamp ASC
    """
    df_data = pd.read_sql_query(query, conn)
    conn.close()
    return df_data

if st.button("🔄 Odśwież dane"):
    st.rerun()

df = load_data(hours_back)

if df.empty:
    st.info(f"Brak danych z ostatnich {hours_back} godzin w bazie.")
else:
    # KOLEKCJA WARTOSCI: Scalanie val_num oraz val_str (dla booleanów typu True/False)
    df["val_combined"] = df["val_num"]
    bool_map = {
        "True": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0,
        "False": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0
    }
    mask_str = df["val_combined"].isna() & df["val_str"].notna()
    df.loc[mask_str, "val_combined"] = df.loc[mask_str, "val_str"].map(bool_map)

    df_pivot = df.pivot_table(index="czas", columns="code", values="val_combined", aggfunc="first").reset_index()
    df_pivot["czas"] = pd.to_datetime(df_pivot["czas"])
    df_pivot = df_pivot.sort_values("czas")

    needed_cols = ["out_water_temp", "in_water_temp", "flow_rate", "ac_vol", "ac_curr", "comp_freq", "disc_temp", "amb_temp", "valve", "heat_temp_set", "defrost"]
    for col in needed_cols:
        if col not in df_pivot.columns:
            df_pivot[col] = np.nan
        else:
            df_pivot[col] = df_pivot[col].ffill()

    # Domyślnie zawór = 0 (CO) jeśli brak danych
    df_pivot["valve"] = df_pivot["valve"].fillna(0).astype(float)

    if resample_rule:
        df_pivot = df_pivot.set_index("czas").resample(resample_rule).agg({
            "out_water_temp": "mean",
            "in_water_temp": "mean",
            "flow_rate": "mean",
            "ac_vol": "mean",
            "ac_curr": "mean",
            "comp_freq": "mean",
            "disc_temp": "mean",
            "amb_temp": "mean",
            "heat_temp_set": "last",
            "valve": "mean",
            "defrost": "max"
        }).reset_index()
        for col in needed_cols:
            df_pivot[col] = df_pivot[col].ffill()

        # PO WŁASCIWYM ZAPISIE: valve >= 0.5 oznacza CWU, poniżej 0.5 to CO
        df_pivot["Tryb"] = np.where(df_pivot["valve"] >= 0.5, "CWU", "CO")
    else:
        df_pivot["Tryb"] = np.where(df_pivot["valve"] >= 0.5, "CWU", "CO")

    # --- OBLICZENIA FIZYCZNE ---
    curr_a = df_pivot["ac_curr"] / ac_curr_div
    df_pivot["flow_m3h"] = df_pivot["flow_rate"] / 10.0
    df_pivot["delta_t"] = df_pivot["out_water_temp"] - df_pivot["in_water_temp"]

    df_pivot["P_th_kw"] = (df_pivot["flow_m3h"] * 4.186 * df_pivot["delta_t"]) / 3.6
    df_pivot["P_el_kw"] = (df_pivot["ac_vol"] * curr_a * cos_phi) / 1000.0
    df_pivot["COP"] = df_pivot["P_th_kw"] / df_pivot["P_el_kw"]

    invalid_mask = (df_pivot["P_el_kw"] < 0.1) | (df_pivot["P_th_kw"] <= 0) | (df_pivot["COP"] < 0.5) | (df_pivot["COP"] > 10.0)
    df_pivot.loc[invalid_mask, "COP"] = np.nan
    df_pivot.loc[df_pivot["P_th_kw"] < 0, "P_th_kw"] = 0.0

    # ENERGIA I SCOP
    df_pivot["dt_hours"] = df_pivot["czas"].diff().dt.total_seconds().fillna(0) / 3600.0
    df_pivot["E_th_kwh"] = df_pivot["P_th_kw"] * df_pivot["dt_hours"]
    df_pivot["E_el_kwh"] = df_pivot["P_el_kw"] * df_pivot["dt_hours"]

    co_mask = (df_pivot["Tryb"] == "CO") & (~df_pivot["COP"].isna())
    cwu_mask = (df_pivot["Tryb"] == "CWU") & (~df_pivot["COP"].isna())

    e_th_co = df_pivot.loc[co_mask, "E_th_kwh"].sum()
    e_el_co = df_pivot.loc[co_mask, "E_el_kwh"].sum()
    scop_co = (e_th_co / e_el_co) if e_el_co > 0 else 0.0

    e_th_cwu = df_pivot.loc[cwu_mask, "E_th_kwh"].sum()
    e_el_cwu = df_pivot.loc[cwu_mask, "E_el_kwh"].sum()
    scop_cwu = (e_th_cwu / e_el_cwu) if e_el_cwu > 0 else 0.0

    e_th_total = e_th_co + e_th_cwu
    e_el_total = e_el_co + e_el_cwu
    scop_total = (e_th_total / e_el_total) if e_el_total > 0 else 0.0

    # WYKRYWANIE CYKLI DEFROST
    df_pivot["defrost_num"] = df_pivot["defrost"].fillna(0).apply(lambda x: 1 if x else 0)
    df_pivot["defrost_start"] = ((df_pivot["defrost_num"] == 1) & (df_pivot["defrost_num"].shift(1, fill_value=0) == 0)).astype(int)

    # AGREGACJA DZIENNA
    df_pivot["dzień"] = df_pivot["czas"].dt.date
    df_pivot["E_el_co_row"] = np.where(df_pivot["Tryb"] == "CO", df_pivot["E_el_kwh"], 0.0)
    df_pivot["E_el_cwu_row"] = np.where(df_pivot["Tryb"] == "CWU", df_pivot["E_el_kwh"], 0.0)
    df_pivot["E_th_co_row"] = np.where(df_pivot["Tryb"] == "CO", df_pivot["E_th_kwh"], 0.0)
    df_pivot["E_th_cwu_row"] = np.where(df_pivot["Tryb"] == "CWU", df_pivot["E_th_kwh"], 0.0)

    daily_df = df_pivot.groupby("dzień").agg({
        "E_el_co_row": "sum",
        "E_el_cwu_row": "sum",
        "E_th_co_row": "sum",
        "E_th_cwu_row": "sum",
        "amb_temp": "mean",
        "defrost_start": "sum"
    }).reset_index()

    daily_df["E_el_total"] = daily_df["E_el_co_row"] + daily_df["E_el_cwu_row"]
    daily_df["E_th_total"] = daily_df["E_th_co_row"] + daily_df["E_th_cwu_row"]
    daily_df["SCOP_dzienny"] = np.where(daily_df["E_el_total"] > 0, daily_df["E_th_total"] / daily_df["E_el_total"], np.nan)

    num_days = max(len(daily_df), 1)
    avg_daily_el_co = daily_df["E_el_co_row"].sum() / num_days
    avg_daily_el_cwu = daily_df["E_el_cwu_row"].sum() / num_days
    avg_amb_temp = df_pivot["amb_temp"].mean()
    total_defrosts = int(daily_df["defrost_start"].sum())

    tab_main, tab_scop, tab_diag = st.tabs(["📊 Panel Główny", "🏆 Bilans Energetyczny & SCOP", "🏥 Diagnostyka Pompy"])

    # ZAKŁADKA 1
    with tab_main:
        latest_df = df.drop_duplicates(subset=["code"], keep="last")
        def get_val(c):
            row = latest_df[latest_df["code"] == c]
            if not row.empty:
                v_num = row["val_num"].values[0]
                if pd.notnull(v_num):
                    return f"{v_num} °C" if "temp" in c or c in ["tidr", "back_temp", "heat_temp_set"] else f"{v_num}"
                return str(row["val_str"].values[0])
            return "N/A"

        latest_cop = df_pivot["COP"].dropna().iloc[-1] if not df_pivot["COP"].dropna().empty else 0.0
        latest_p_th = df_pivot["P_th_kw"].iloc[-1] if not df_pivot.empty else 0.0
        latest_p_el = df_pivot["P_el_kw"].iloc[-1] if not df_pivot.empty else 0.0
        latest_flow = df_pivot["flow_m3h"].iloc[-1] if not df_pivot.empty else 0.0
        current_mode = df_pivot["Tryb"].iloc[-1] if not df_pivot.empty else "CO"

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Woda CWU", get_val("tank_temp"))
        c2.metric("Powrót CO", get_val("in_water_temp"))
        c3.metric("Zasilanie CO", get_val("out_water_temp"))
        c4.metric("🎯 Nastawa CO", get_val("heat_temp_set"))
        c5.metric("Przepływ", f"{latest_flow:.1f} m³/h", delta=f"{latest_flow * 1000 / 60:.1f} L/min")
        c6.metric("📊 Chwilowe COP", f"{latest_cop:.2f}", delta=f"Tryb: {current_mode}")

        cp1, cp2 = st.columns(2)
        cp1.metric("🔥 Moc cieplna (P_th)", f"{latest_p_th:.2f} kW")
        cp2.metric("⚡ Pobór prądu (P_el)", f"{latest_p_el:.2f} kW")

        st.markdown("---")

        st.subheader("📊 Chwilowe COP z podziałem na tryb CO / CWU")
        fig_cop = px.line(
            df_pivot.dropna(subset=["COP"]),
            x="czas", y="COP", color="Tryb",
            color_discrete_map={"CO": "#2ECC71", "CWU": "#E67E22"},
            title="Wykres chwilowego COP (Zielony = CO, Pomarańczowy = CWU)",
            markers=(resample_rule is not None)
        )
        fig_cop.update_layout(hovermode="x unified")
        st.plotly_chart(fig_cop, width="stretch")

        st.subheader("📈 Przebieg wybranych parametrów")
        all_codes = df["code"].unique().tolist()
        default_temps = [c for c in ["tank_temp", "in_water_temp", "out_water_temp", "heat_temp_set", "amb_temp"] if c in all_codes]
        selected_temps = st.multiselect("Wybierz parametry do wyświetlenia:", options=all_codes, default=default_temps, format_func=get_param_label)

        if selected_temps:
            temp_df = df[df["code"].isin(selected_temps) & df["val_num"].notnull()].copy()
            if resample_rule:
                temp_df["czas"] = pd.to_datetime(temp_df["czas"])
                temp_df = temp_df.groupby(["code", pd.Grouper(key="czas", freq=resample_rule)])["val_num"].mean().reset_index()

            temp_df["Parametr"] = temp_df["code"].map(lambda c: PARAM_INFO.get(c, {}).get("label", c))
            temp_df["Opis"] = temp_df["code"].map(lambda c: PARAM_INFO.get(c, {}).get("desc", "Brak opisu"))

            fig_temp = px.line(
                temp_df, x="czas", y="val_num", color="Parametr",
                hover_data={"Parametr": True, "Opis": True, "val_num": ":.1f", "code": False},
                title="Wykres wartości parametrów w czasie"
            )
            fig_temp.update_layout(hovermode="x unified")
            st.plotly_chart(fig_temp, width="stretch")

    # ZAKŁADKA 2: BILANS ENERGETYCZNY & SCOP
    with tab_scop:
        st.header("🏆 Podsumowanie Efektywności SCOP i Zużycia Energii")
        
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        sc_col1.metric("🌟 SCOP Całkowite", f"{scop_total:.2f}")
        sc_col2.metric("🏠 SCOP dla CO (Ogrzewanie)", f"{scop_co:.2f}")
        sc_col3.metric("🚿 SCOP dla CWU (Ciepła Woda)", f"{scop_cwu:.2f}")

        st.markdown("### 📊 Statystyki Średniodobowe i Odszranianie")
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("⚡ Śr. dzienne zużycie CO", f"{avg_daily_el_co:.2f} kWh/dzień")
        d_col2.metric("⚡ Śr. dzienne zużycie CWU", f"{avg_daily_el_cwu:.2f} kWh/dzień")
        d_col3.metric("🌡️ Średniodobowa temp. zewn.", f"{avg_amb_temp:.1f} °C" if not np.isnan(avg_amb_temp) else "Brak danych")
        d_col4.metric("❄️ Liczba defrostów (okres)", f"{total_defrosts}")

        st.markdown("---")
        st.subheader("⚡ Zużycie Prądu i Wygenerowane Ciepło [kWh] (Całkowite)")
        
        summary_data = {
            "Obieg / Tryb": ["🏠 Ogrzewanie (CO)", "🚿 Ciepła Woda (CWU)", " TOTAL (Łącznie)"],
            "Pobrana Energia El. [kWh]": [f"{e_el_co:.2f}", f"{e_el_cwu:.2f}", f"{e_el_total:.2f}"],
            "Oddane Ciepło [kWh]": [f"{e_th_co:.2f}", f"{e_th_cwu:.2f}", f"{e_th_total:.2f}"],
            "Średnie SCOP": [f"{scop_co:.2f}", f"{scop_cwu:.2f}", f"{scop_total:.2f}"]
        }
        st.table(pd.DataFrame(summary_data))

        fig_bar = go.Figure(data=[
            go.Bar(name='Prąd pobrany [kWh]', x=['Ogrzewanie CO', 'Ciepła Woda CWU'], y=[e_el_co, e_el_cwu], marker_color='#3498DB'),
            go.Bar(name='Ciepło oddane [kWh]', x=['Ogrzewanie CO', 'Ciepła Woda CWU'], y=[e_th_co, e_th_cwu], marker_color='#E74C3C')
        ])
        fig_bar.update_layout(barmode='group', title="Porównanie energii pobranej do oddanej według trybu pracy")
        st.plotly_chart(fig_bar, width="stretch")

        st.markdown("---")
        st.subheader("📅 Dzienny Bilans Zużycia, Temperatur i Defrostów")
        daily_display = daily_df[["dzień", "amb_temp", "E_el_co_row", "E_el_cwu_row", "E_el_total", "defrost_start", "SCOP_dzienny"]].copy()
        daily_display.columns = ["Data", "Śr. Temp Zewn. [°C]", "Prąd CO [kWh]", "Prąd CWU [kWh]", "Prąd Łącznie [kWh]", "Liczba Defrostów", "SCOP Dzienny"]
        daily_display["Śr. Temp Zewn. [°C]"] = daily_display["Śr. Temp Zewn. [°C]"].round(1)
        daily_display["Prąd CO [kWh]"] = daily_display["Prąd CO [kWh]"].round(2)
        daily_display["Prąd CWU [kWh]"] = daily_display["Prąd CWU [kWh]"].round(2)
        daily_display["Prąd Łącznie [kWh]"] = daily_display["Prąd Łącznie [kWh]"].round(2)
        daily_display["SCOP Dzienny"] = daily_display["SCOP Dzienny"].round(2)
        st.dataframe(daily_display, width="stretch", hide_index=True)

    # ZAKŁADKA 3: DIAGNOSTYKA
    with tab_diag:
        st.header("🏥 Centrum Diagnostyczne Pompy Ciepła")
        st.subheader("⚠️ Status Pracy i Ostrzeżenia")
        col_a1, col_a2, col_a3 = st.columns(3)

        last_disc = df_pivot["disc_temp"].dropna().iloc[-1] if not df_pivot["disc_temp"].dropna().empty else None
        with col_a1:
            if last_disc and last_disc >= 90.0:
                st.error(f"🔴 **KRYTYCZNA TEMP. TŁOCZENIA:** {last_disc:.1f}°C\nRyzyko przegrzania sprężarki!")
            elif last_disc and last_disc >= 80.0:
                st.warning(f"🟡 **Podwyższona temp. tłoczenia:** {last_disc:.1f}°C")
            elif last_disc:
                st.success(f"🟢 **Temp. tłoczenia w normie:** {last_disc:.1f}°C")
            else:
                st.info("⚪ Brak danych temp. tłoczenia")

        last_dt = df_pivot["delta_t"].dropna().iloc[-1] if not df_pivot["delta_t"].dropna().empty else None
        is_pumping = df_pivot["P_el_kw"].iloc[-1] > 0.2 if not df_pivot.empty else False
        with col_a2:
            if is_pumping and last_dt is not None:
                if last_dt < 2.0:
                    st.warning(f"🟡 **Za małe ΔT ({last_dt:.1f}°C):** Przepływ wody za duży lub brak odbioru ciepła.")
                elif last_dt > 8.0:
                    st.warning(f"🟡 **Za duże ΔT ({last_dt:.1f}°C):** Zbyt mały przepływ wody (sprawdź pompę/filtry).")
                else:
                    st.success(f"🟢 **Różnica ΔT w normie:** {last_dt:.1f}°C (Idealnie: 3-6°C)")
            else:
                st.info("⚪ Pompa w stanie spoczynku (ΔT pauza)")

        is_comp_on = df_pivot["comp_freq"] > 5
        starts_count = (is_comp_on & (~is_comp_on.shift(1, fill_value=False))).sum()
        with col_a3:
            if starts_count > 15:
                st.warning(f"🟡 **Wykryto taktowanie!** Liczba startów sprężarki: **{starts_count}** w wybranym oknie.")
            else:
                st.success(f"🟢 **Cykliczność w normie:** Liczba startów sprężarki: **{starts_count}**")

        st.markdown("---")

        st.subheader("1️⃣ Odbiór ciepła przez instalację (Różnica temperatur ΔT)")
        fig_dt = go.Figure()
        fig_dt.add_trace(go.Scatter(x=df_pivot["czas"], y=df_pivot["delta_t"], mode='lines', name='Różnica ΔT (°C)', line=dict(color='#3498DB', width=2)))
        fig_dt.add_hrect(y0=3.0, y1=7.0, fillcolor="Green", opacity=0.15, line_width=0, annotation_text="Strefa optymalna (3 - 7 °C)", annotation_position="top left")
        fig_dt.update_layout(hovermode="x unified", xaxis_title="Czas", yaxis_title="ΔT (°C)")
        st.plotly_chart(fig_dt, width="stretch")

        st.subheader("2️⃣ Bezpieczeństwo Sprężarki (Temperatura Tłoczenia Discharge)")
        fig_disc = go.Figure()
        fig_disc.add_trace(go.Scatter(x=df_pivot["czas"], y=df_pivot["disc_temp"], mode='lines', name='Temp. Tłoczenia (°C)', line=dict(color='#E67E22', width=2)))
        fig_disc.add_trace(go.Scatter(x=df_pivot["czas"], y=df_pivot["comp_freq"], mode='lines', name='Obroty sprężarki (Hz)', line=dict(color='#9B59B6', width=1.5, dash='dot')))
        fig_disc.add_hline(y=90.0, line_dash="dash", line_color="Red", annotation_text="Krytyczne 90°C", annotation_position="bottom right")
        fig_disc.update_layout(hovermode="x unified", xaxis_title="Czas", yaxis_title="Wartość")
        st.plotly_chart(fig_disc, width="stretch")