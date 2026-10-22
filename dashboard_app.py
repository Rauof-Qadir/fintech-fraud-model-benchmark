
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PROJECT PATHS & WHITE THEME
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORT_DIR = ROOT / "reports"

DATA_CANDIDATES = [
    RAW_DIR / "transactions.csv",
    RAW_DIR / "PS_20174392719_1491204439457_log.csv",
    PROCESSED_DIR / "transactions_clean.csv",
    PROCESSED_DIR / "transactions.csv",
]

st.set_page_config(
    page_title="Fintelligence | Fraud Risk Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional light palette
BG = "#FFFFFF"
PANEL = "#FFFFFF"
BORDER = "#E2E8F0"
CYAN = "#0891B2"
GREEN = "#059669"
RED = "#DC2626"
AMBER = "#D97706"
PURPLE = "#7C3AED"
TEXT = "#172033"
MUTED = "#64748B"
GRID = "#E9EEF5"
BLUE = "#2563EB"

CHART_COLORS = [
    BLUE, CYAN, GREEN, PURPLE, AMBER, RED,
    "#DB2777", "#475569"
]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'Manrope', sans-serif;
        color: {TEXT};
    }}

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {{
        background: {BG} !important;
        color: {TEXT} !important;
    }}

    [data-testid="stSidebar"] {{
        background: #F8FAFC !important;
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] * {{
        color: {TEXT};
    }}

    [data-testid="stHeader"] {{
        background: rgba(255,255,255,0.96) !important;
    }}

    h1, h2, h3, h4, p, label,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"] {{
        color: {TEXT};
    }}

    h1, h2, h3 {{
        letter-spacing: -0.035em;
    }}

    .hero {{
        padding: 30px;
        border: 1px solid {BORDER};
        border-radius: 20px;
        background: linear-gradient(120deg, #FFFFFF 0%, #F0F9FF 60%, #F5F3FF 100%);
        margin-bottom: 24px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.045);
    }}

    .eyebrow {{
        color: {CYAN} !important;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: 11px;
        font-weight: 800;
    }}

    .hero-title {{
        color: {TEXT} !important;
        font-size: clamp(28px, 3vw, 43px);
        font-weight: 800;
        line-height: 1.18;
        margin-top: 10px;
    }}

    .hero-sub {{
        color: {MUTED} !important;
        font-size: 14px;
        margin-top: 12px;
        line-height: 1.8;
    }}

    .metric-card {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 19px;
        min-height: 116px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.035);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}

    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }}

    .metric-label {{
        color: {MUTED} !important;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .09em;
        font-weight: 800;
    }}

    .metric-value {{
        font-size: clamp(22px, 2vw, 30px);
        font-weight: 800;
        margin-top: 10px;
        overflow-wrap: anywhere;
    }}

    .metric-foot {{
        color: {MUTED} !important;
        font-size: 11px;
        margin-top: 6px;
    }}

    .note {{
        background: #F0F9FF;
        color: {TEXT} !important;
        border: 1px solid #BAE6FD;
        border-left: 4px solid {CYAN};
        border-radius: 12px;
        padding: 16px;
        line-height: 1.8;
        font-size: 13px;
    }}

    .warning {{
        background: #FFFBEB;
        color: #78350F !important;
        border: 1px solid #FDE68A;
        border-left: 4px solid {AMBER};
        border-radius: 12px;
        padding: 16px;
        line-height: 1.8;
        font-size: 13px;
    }}

    .warning b, .note b {{
        color: inherit !important;
    }}

    div[data-testid="stPlotlyChart"] {{
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 8px;
        background: #FFFFFF;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.025);
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {{
        background-color: #FFFFFF;
        border-color: {BORDER};
    }}

    [data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 14px;
    }}

    hr {{
        border-color: {BORDER} !important;
    }}

    button[kind="secondary"] {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. FILE HELPERS
# ============================================================
def find_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def find_report(*filenames):
    for filename in filenames:
        path = REPORT_DIR / filename
        if path.exists():
            return path
    return None


def load_csv(path):
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def load_json(path):
    if path is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            result = json.load(file)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


def find_column(df, *names):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def numeric_series(df, *names):
    column = find_column(df, *names)
    if column is None:
        return None
    return pd.to_numeric(df[column], errors="coerce")


# ============================================================
# 3. GLOBAL PLOTLY WHITE STYLE
# ============================================================
def style_chart(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            color=TEXT,
            family="Manrope, sans-serif",
            size=12,
        ),
        title=dict(
            font=dict(size=16, color=TEXT),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=18, r=18, t=65, b=25),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=BORDER,
            borderwidth=0,
            font=dict(color=TEXT, size=11),
            orientation="h",
            y=1.12,
            x=0,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=BORDER,
            font=dict(color=TEXT, family="Manrope, sans-serif"),
        ),
    )

    fig.update_xaxes(
        gridcolor=GRID,
        linecolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=TEXT),
        showline=True,
        automargin=True,
    )

    fig.update_yaxes(
        gridcolor=GRID,
        linecolor=BORDER,
        zerolinecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title_font=dict(color=TEXT),
        showline=False,
        automargin=True,
    )

    return fig


def metric_card(label, value, description, color=CYAN):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{value}</div>
            <div class="metric-foot">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 4. SCAN THE FULL CSV IN CHUNKS
# ============================================================
@st.cache_data(show_spinner="Scanning your actual transaction dataset...")
def scan_transactions(path_string, modified_time):
    path = Path(path_string)

    total_rows = 0
    fraud_count = 0
    amount_sum = 0.0
    amount_count = 0

    type_counts = {}
    type_fraud = {}
    time_counts = {}

    samples = []
    sample_limit = 100_000

    for chunk in pd.read_csv(
        path,
        chunksize=250_000,
        low_memory=False,
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        total_rows += len(chunk)

        fraud_col = find_column(chunk, "isFraud")
        amount_col = find_column(chunk, "amount")
        type_col = find_column(chunk, "type")
        step_col = find_column(chunk, "step")

        if fraud_col:
            fraud = (
                pd.to_numeric(chunk[fraud_col], errors="coerce")
                .fillna(0)
            )
            fraud_count += int(fraud.sum())
        else:
            fraud = pd.Series(0, index=chunk.index)

        if amount_col:
            amount = pd.to_numeric(chunk[amount_col], errors="coerce")
            amount_sum += float(amount.sum())
            amount_count += int(amount.notna().sum())

        if type_col:
            types = chunk[type_col].fillna("Unknown").astype(str)

            for name, count in types.value_counts().items():
                type_counts[name] = type_counts.get(name, 0) + int(count)

            if fraud_col:
                grouped = pd.DataFrame({
                    "type": types,
                    "fraud": fraud.values,
                }).groupby("type")["fraud"].sum()

                for name, count in grouped.items():
                    type_fraud[name] = (
                        type_fraud.get(name, 0) + int(count)
                    )

        if step_col and fraud_col:
            steps = pd.to_numeric(chunk[step_col], errors="coerce")
            temp = pd.DataFrame({
                "step": steps,
                "fraud": fraud.values,
            }).dropna()

            grouped = temp.groupby("step").agg(
                transactions=("fraud", "size"),
                fraud=("fraud", "sum"),
            )

            for step, row in grouped.iterrows():
                key = int(step)
                previous = time_counts.get(key, [0, 0])
                time_counts[key] = [
                    previous[0] + int(row["transactions"]),
                    previous[1] + int(row["fraud"]),
                ]

        if len(samples) < sample_limit:
            remaining = sample_limit - len(samples)
            samples.extend(chunk.head(remaining).to_dict("records"))

    sample_df = pd.DataFrame(samples)

    stats = {
        "transactions": total_rows,
        "fraud": fraud_count,
        "fraud_rate": fraud_count / total_rows if total_rows else 0,
        "average_amount": (
            amount_sum / amount_count if amount_count else None
        ),
        "type_counts": type_counts,
        "type_fraud": type_fraud,
        "time_counts": time_counts,
        "sample_size": len(sample_df),
        "columns": list(sample_df.columns),
    }

    return stats, sample_df


# ============================================================
# 5. LOAD REPORTS FROM YOUR PROJECT
# ============================================================
dataset_path = find_dataset()

test_report_path = find_report("test_evaluation.csv")
benchmark_path = find_report("model_benchmark.csv")
threshold_path = find_report(
    "threshold_benchmark.csv",
    "threshold_analysis.csv",
)
cost_path = find_report(
    "final_cost_aware_model_audit.csv",
    "all_strategy_cost_sensitivity.csv",
    "cost_sensitivity_analysis.csv",
    "best_strategy_by_cost_scenario.csv",
)
ablation_path = find_report(
    "controlled_ablation.csv",
    "feature_ablation.csv",
)
origin_ablation_path = find_report("origin_feature_ablation.csv")
importance_path = find_report("feature_importance.csv")
split_path = find_report("split_report.json")
quality_path = find_report("data_quality_report.json")

test_report = load_csv(test_report_path)
benchmark_report = load_csv(benchmark_path)
threshold_report = load_csv(threshold_path)
cost_report = load_csv(cost_path)
ablation_report = load_csv(ablation_path)
origin_report = load_csv(origin_ablation_path)
importance_report = load_csv(importance_path)
split_report = load_json(split_path)
quality_report = load_json(quality_path)

model_report = test_report.copy()

if (
    model_report.empty
    or find_column(model_report, "model", "model_name") is None
):
    model_report = benchmark_report.copy()

if not model_report.empty:
    aliases = {
        "Model": ["model", "model_name"],
        "PR-AUC": [
            "pr_auc",
            "average_precision",
            "average_precision_pr_auc",
        ],
        "ROC-AUC": ["roc_auc"],
        "Precision": ["precision"],
        "Recall": ["recall"],
        "F1": ["f1", "f1_score"],
        "Threshold": ["threshold", "selected_threshold"],
    }

    rename = {}
    for standard_name, possible_names in aliases.items():
        original = find_column(model_report, *possible_names)
        if original is not None and original != standard_name:
            rename[original] = standard_name

    model_report.rename(columns=rename, inplace=True)


# ============================================================
# 6. SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🛡️ FINTELLIGENCE")
    st.caption("FRAUD RISK INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Model Performance",
            "Transaction Explorer",
            "Risk & Cost Lab",
            "Temporal Drift",
            "Feature Risk Audit",
            "Data Sources",
        ],
    )

    st.divider()
    st.markdown("### Data status")

    st.write(
        ("🟢" if dataset_path else "🔴")
        + " Transaction CSV "
        + ("found" if dataset_path else "missing")
    )
    st.write(
        ("🟢" if not model_report.empty else "🟡")
        + " Model report "
        + ("loaded" if not model_report.empty else "missing")
    )
    st.write(
        ("🟢" if not cost_report.empty else "🟡")
        + " Cost report "
        + ("loaded" if not cost_report.empty else "missing")
    )

    st.caption("Sources: your local project files.")


# ============================================================
# 7. APP HEADER
# ============================================================
st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Risk intelligence / model observability</div>
        <div class="hero-title">
            Fraud Detection Command Center
            <span style="color:{CYAN}">.</span>
        </div>
        <div class="hero-sub">
            Transaction intelligence, model benchmarking, cost analysis,
            temporal drift and feature-risk auditing — connected to your
            existing project files.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = {}
transactions = pd.DataFrame()

if dataset_path:
    stats, transactions = scan_transactions(
        str(dataset_path),
        dataset_path.stat().st_mtime,
    )


# ============================================================
# PAGE 1 — EXECUTIVE OVERVIEW
# ============================================================
if page == "Executive Overview":
    st.subheader("Executive snapshot")

    if stats:
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Transactions",
                f"{stats['transactions']:,}",
                "Full CSV scanned",
                CYAN,
            )

        with c2:
            metric_card(
                "Fraud cases",
                f"{stats['fraud']:,}",
                "Fraud-labelled rows",
                RED,
            )

        with c3:
            metric_card(
                "Fraud rate",
                f"{stats['fraud_rate']:.4%}",
                "Across full dataset",
                AMBER,
            )

        with c4:
            amount = stats["average_amount"]
            amount_text = (
                f"{amount:,.2f}" if amount is not None else "N/A"
            )
            metric_card(
                "Average amount",
                amount_text,
                "Dataset units, not assumed currency",
                PURPLE,
            )
    else:
        st.warning(
            "Transaction CSV not found. Check the Data Sources page "
            "and place the file in data/raw/."
        )

    left, right = st.columns(2)

    with left:
        if (
            not model_report.empty
            and {"Model", "PR-AUC"}.issubset(model_report.columns)
        ):
            plot = model_report.copy()
            plot["PR-AUC"] = pd.to_numeric(
                plot["PR-AUC"], errors="coerce"
            )
            plot = plot.dropna(subset=["PR-AUC"])

            if not plot.empty:
                fig = px.bar(
                    plot,
                    x="Model",
                    y="PR-AUC",
                    color="Model",
                    text=plot["PR-AUC"].map(lambda x: f"{x:.4f}"),
                    title="Saved model comparison · PR-AUC",
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_traces(
                    textposition="outside",
                    textfont=dict(color=TEXT, size=11),
                    cliponaxis=False,
                )
                st.plotly_chart(
                    style_chart(fig, 370),
                    use_container_width=True,
                )
        else:
            st.info(
                "Model chart needs a report with Model and PR-AUC columns."
            )

    with right:
        if stats.get("type_counts"):
            type_df = pd.DataFrame({
                "Transaction type": list(stats["type_counts"].keys()),
                "Transactions": list(stats["type_counts"].values()),
            })

            fig = px.pie(
                type_df,
                names="Transaction type",
                values="Transactions",
                hole=0.60,
                title="Transaction mix · full dataset",
                color_discrete_sequence=CHART_COLORS,
            )
            fig.update_traces(
                textinfo="percent",
                textfont=dict(color=TEXT, size=11),
                marker=dict(line=dict(color="#FFFFFF", width=2)),
            )
            st.plotly_chart(
                style_chart(fig, 370),
                use_container_width=True,
            )
        else:
            st.info("Transaction type distribution is unavailable.")

    if stats.get("time_counts"):
        time_df = pd.DataFrame([
            {
                "Step": step,
                "Transactions": values[0],
                "Fraud": values[1],
                "Fraud rate": values[1] / values[0] if values[0] else 0,
            }
            for step, values in sorted(stats["time_counts"].items())
        ])

        fig = px.line(
            time_df,
            x="Step",
            y="Fraud rate",
            title="Fraud prevalence across time steps",
            color_discrete_sequence=[RED],
        )
        fig.update_traces(line=dict(width=2.5), mode="lines")
        fig.update_yaxes(tickformat=".2%")

        st.plotly_chart(
            style_chart(fig, 370),
            use_container_width=True,
        )

    if not model_report.empty:
        st.markdown("### Saved model report")
        st.dataframe(
            model_report,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <div class="warning">
        <b>Research limitation:</b> this project uses synthetic transaction
        data. Previously explored test results are not a pristine,
        untouched holdout and must not be interpreted as proven live-bank
        performance.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 2 — MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    st.subheader("Model performance explorer")

    if model_report.empty:
        st.warning(
            "No model report found. Check reports/test_evaluation.csv "
            "and reports/model_benchmark.csv."
        )
    else:
        st.dataframe(
            model_report,
            use_container_width=True,
            hide_index=True,
        )

        metrics = [
            metric for metric in
            ["PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"]
            if metric in model_report.columns
        ]

        if metrics and "Model" in model_report.columns:
            selected_models = st.multiselect(
                "Models to compare",
                model_report["Model"].astype(str).tolist(),
                default=model_report["Model"].astype(str).tolist(),
            )

            selected_metric = st.selectbox(
                "Metric",
                metrics,
                index=0,
            )

            plot = model_report[
                model_report["Model"].astype(str).isin(selected_models)
            ].copy()

            plot[selected_metric] = pd.to_numeric(
                plot[selected_metric], errors="coerce"
            )
            plot = plot.dropna(subset=[selected_metric])

            if not plot.empty:
                fig = px.bar(
                    plot,
                    x="Model",
                    y=selected_metric,
                    color="Model",
                    text=plot[selected_metric].map(lambda x: f"{x:.4f}"),
                    title=f"{selected_metric} by model",
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_traces(
                    textposition="outside",
                    textfont=dict(color=TEXT),
                    cliponaxis=False,
                )
                st.plotly_chart(
                    style_chart(fig, 400),
                    use_container_width=True,
                )

    st.markdown(
        """
        <div class="note">
        <b>How to interpret:</b> PR-AUC is useful for highly imbalanced
        fraud data. Precision measures how many flagged transactions
        are fraud; recall measures how much fraud the model catches.
        Read them together.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 3 — TRANSACTION EXPLORER
# ============================================================
elif page == "Transaction Explorer":
    st.subheader("Transaction explorer")

    if transactions.empty:
        st.warning(
            "No dataset sample available. Add the original CSV under data/raw/."
        )
    else:
        st.caption(
            f"Interactive charts use up to {len(transactions):,} rows "
            "from the beginning of the local CSV. Full-dataset counts "
            "are shown on the Overview page."
        )

        type_col = find_column(transactions, "type")
        amount_col = find_column(transactions, "amount")
        fraud_col = find_column(transactions, "isFraud")
        step_col = find_column(transactions, "step")

        filtered = transactions.copy()

        if type_col:
            options = sorted(
                filtered[type_col].dropna().astype(str).unique()
            )
            selected_types = st.multiselect(
                "Transaction types",
                options,
                default=options,
            )
            filtered = filtered[
                filtered[type_col].astype(str).isin(selected_types)
            ]

        if fraud_col:
            label_filter = st.selectbox(
                "Fraud label",
                ["All", "Fraud only", "Non-fraud only"],
            )

            fraud_values = pd.to_numeric(
                filtered[fraud_col], errors="coerce"
            )

            if label_filter == "Fraud only":
                filtered = filtered[fraud_values == 1]
            elif label_filter == "Non-fraud only":
                filtered = filtered[fraud_values == 0]

        if amount_col and not filtered.empty:
            amounts = pd.to_numeric(
                filtered[amount_col], errors="coerce"
            ).dropna()

            if not amounts.empty:
                low = float(amounts.min())
                high = float(amounts.quantile(0.99))

                if high > low:
                    amount_range = st.slider(
                        "Amount range (up to sample 99th percentile)",
                        min_value=low,
                        max_value=high,
                        value=(low, high),
                    )

                    numeric_amounts = pd.to_numeric(
                        filtered[amount_col], errors="coerce"
                    )
                    filtered = filtered[
                        numeric_amounts.between(
                            amount_range[0], amount_range[1]
                        )
                    ]

        left, right = st.columns(2)

        with left:
            if type_col and fraud_col and not filtered.empty:
                chart_data = filtered.copy()
                chart_data["_label"] = (
                    pd.to_numeric(
                        chart_data[fraud_col], errors="coerce"
                    )
                    .map({0: "Non-fraud", 1: "Fraud"})
                    .fillna("Unknown")
                )

                grouped = (
                    chart_data.groupby([type_col, "_label"])
                    .size()
                    .reset_index(name="Transactions")
                )

                fig = px.bar(
                    grouped,
                    x=type_col,
                    y="Transactions",
                    color="_label",
                    barmode="group",
                    title="Transaction type vs fraud label",
                    color_discrete_map={
                        "Fraud": RED,
                        "Non-fraud": CYAN,
                        "Unknown": PURPLE,
                    },
                )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True,
                )

        with right:
            if amount_col and not filtered.empty:
                chart_data = filtered.copy()
                chart_data["_amount"] = pd.to_numeric(
                    chart_data[amount_col], errors="coerce"
                )

                if fraud_col:
                    chart_data["_label"] = (
                        pd.to_numeric(
                            chart_data[fraud_col], errors="coerce"
                        )
                        .map({0: "Non-fraud", 1: "Fraud"})
                        .fillna("Unknown")
                    )
                    fig = px.histogram(
                        chart_data,
                        x="_amount",
                        color="_label",
                        nbins=45,
                        title="Transaction amount distribution",
                        color_discrete_map={
                            "Non-fraud": CYAN,
                            "Fraud": RED,
                            "Unknown": PURPLE,
                        },
                    )
                else:
                    fig = px.histogram(
                        chart_data,
                        x="_amount",
                        nbins=45,
                        title="Transaction amount distribution",
                        color_discrete_sequence=[CYAN],
                    )

                st.plotly_chart(
                    style_chart(fig),
                    use_container_width=True,
                )

        if step_col and fraud_col and not filtered.empty:
            chart_data = filtered.copy()
            chart_data["_step"] = pd.to_numeric(
                chart_data[step_col], errors="coerce"
            )
            chart_data["_fraud"] = pd.to_numeric(
                chart_data[fraud_col], errors="coerce"
            )

            by_step = (
                chart_data.groupby("_step")
                .agg(
                    Transactions=("_fraud", "size"),
                    Fraud=("_fraud", "sum"),
                )
                .reset_index()
            )
            by_step["Fraud rate"] = (
                by_step["Fraud"] / by_step["Transactions"]
            )

            fig = px.line(
                by_step,
                x="_step",
                y="Fraud rate",
                title="Fraud rate in filtered sample",
                color_discrete_sequence=[RED],
            )
            fig.update_traces(line=dict(width=2.5))
            fig.update_yaxes(tickformat=".2%")

            st.plotly_chart(
                style_chart(fig),
                use_container_width=True,
            )

        st.markdown("### Filtered transactions")
        st.dataframe(
            filtered.head(500),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download displayed sample",
            data=filtered.head(500).to_csv(index=False),
            file_name="filtered_transactions_sample.csv",
            mime="text/csv",
        )


# ============================================================
# PAGE 4 — RISK & COST LAB
# ============================================================
elif page == "Risk & Cost Lab":
    st.subheader("Saved risk and cost experiments")

    st.caption(
        "This page reads experiment results already saved in your reports "
        "folder. It does not fabricate real-world bank costs."
    )

    if cost_report.empty:
        st.warning(
            "No cost report found. Expected final_cost_aware_model_audit.csv, "
            "all_strategy_cost_sensitivity.csv or cost_sensitivity_analysis.csv."
        )
    else:
        st.markdown("### Cost sensitivity report")
        st.dataframe(
            cost_report,
            use_container_width=True,
            hide_index=True,
        )

        numeric_cols = cost_report.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_cols:
            metric_name = st.selectbox(
                "Numeric column to visualize",
                numeric_cols,
            )

            group_col = find_column(
                cost_report,
                "model",
                "strategy",
                "scenario",
                "model_name",
                "strategy_name",
            )

            if group_col:
                plot = cost_report.copy()
                plot[metric_name] = pd.to_numeric(
                    plot[metric_name], errors="coerce"
                )
                plot = plot.dropna(subset=[metric_name])

                fig = px.bar(
                    plot,
                    x=group_col,
                    y=metric_name,
                    color=group_col,
                    title=f"{metric_name} by {group_col}",
                    color_discrete_sequence=CHART_COLORS,
                )
                st.plotly_chart(
                    style_chart(fig, 400),
                    use_container_width=True,
                )

    if not threshold_report.empty:
        st.markdown("### Threshold experiments")
        st.dataframe(
            threshold_report,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Threshold report is not available.")

    st.markdown(
        """
        <div class="warning">
        Cost units and loss multipliers are illustrative unless calibrated
        against real operational data. Do not label these values as actual
        PKR or USD bank losses.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 5 — TEMPORAL DRIFT
# ============================================================
elif page == "Temporal Drift":
    st.subheader("Temporal drift monitor")

    if split_report:
        st.markdown("### Saved split report")
        st.json(split_report, expanded=True)
    else:
        st.info("reports/split_report.json was not found.")

    if stats.get("time_counts"):
        time_df = pd.DataFrame([
            {
                "Step": step,
                "Transactions": values[0],
                "Fraud": values[1],
                "Fraud rate": values[1] / values[0] if values[0] else 0,
            }
            for step, values in sorted(stats["time_counts"].items())
        ])

        fig = px.line(
            time_df,
            x="Step",
            y="Fraud rate",
            title="Fraud rate across all time steps",
            color_discrete_sequence=[RED],
        )
        fig.update_traces(line=dict(width=2.5))
        fig.update_yaxes(tickformat=".2%")

        st.plotly_chart(
            style_chart(fig, 370),
            use_container_width=True,
        )

        fig = px.line(
            time_df,
            x="Step",
            y="Transactions",
            title="Transaction volume across time steps",
            color_discrete_sequence=[BLUE],
        )
        fig.update_traces(line=dict(width=2.5))

        st.plotly_chart(
            style_chart(fig, 320),
            use_container_width=True,
        )
    else:
        st.warning(
            "Temporal analysis requires the step and isFraud columns."
        )

    st.markdown(
        """
        <div class="warning">
        A changing fraud rate indicates possible distribution shift, but
        does not establish its cause. Validate with fresh future-period
        data before claiming production generalization.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 6 — FEATURE RISK AUDIT
# ============================================================
elif page == "Feature Risk Audit":
    st.subheader("Feature dependency audit")

    reports = [
        ("Controlled feature ablation", ablation_report),
        ("Origin-feature ablation", origin_report),
        ("Feature importance", importance_report),
    ]

    found_any = False

    for title, report in reports:
        if report.empty:
            continue

        found_any = True
        st.markdown(f"### {title}")
        st.dataframe(
            report,
            use_container_width=True,
            hide_index=True,
        )

        score_col = find_column(
            report, "pr_auc", "average_precision", "PR-AUC"
        )
        experiment_col = find_column(
            report,
            "experiment",
            "feature_set",
            "feature_group",
            "scenario",
            "feature",
            "feature_name",
        )
        model_col = find_column(report, "model", "model_name")

        if score_col and experiment_col:
            chart_data = report.copy()
            chart_data[score_col] = pd.to_numeric(
                chart_data[score_col], errors="coerce"
            )
            chart_data = chart_data.dropna(subset=[score_col])

            if not chart_data.empty:
                fig = px.bar(
                    chart_data,
                    x=experiment_col,
                    y=score_col,
                    color=model_col if model_col else None,
                    barmode="group",
                    title=f"{title} · {score_col}",
                    color_discrete_sequence=CHART_COLORS,
                )
                fig.update_xaxes(tickangle=-20)
                st.plotly_chart(
                    style_chart(fig, 400),
                    use_container_width=True,
                )

        feature_col = find_column(
            report, "feature", "feature_name", "column"
        )
        numeric_cols = report.select_dtypes(
            include="number"
        ).columns.tolist()

        if feature_col and numeric_cols:
            importance_col = st.selectbox(
                f"Importance value · {title}",
                numeric_cols,
                key=f"importance_{title}",
            )

            plot = report.copy()
            plot[importance_col] = pd.to_numeric(
                plot[importance_col], errors="coerce"
            )
            plot = plot.dropna(subset=[importance_col]).nlargest(
                20, importance_col
            )

            if not plot.empty:
                fig = px.bar(
                    plot.sort_values(importance_col),
                    x=importance_col,
                    y=feature_col,
                    orientation="h",
                    title=f"Top features · {title}",
                    color_discrete_sequence=[PURPLE],
                )
                st.plotly_chart(
                    style_chart(fig, 420),
                    use_container_width=True,
                )

    if not found_any:
        st.warning(
            "No feature reports found. Check reports/feature_ablation.csv, "
            "reports/controlled_ablation.csv, "
            "reports/origin_feature_ablation.csv and "
            "reports/feature_importance.csv."
        )

    st.markdown(
        """
        <div class="warning">
        Strong reliance on balance-related features is a reason to investigate
        feature availability and synthetic-data artifacts. It does not by
        itself prove target leakage.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE 7 — DATA SOURCES
# ============================================================
elif page == "Data Sources":
    st.subheader("Data source and artifact status")

    source_paths = [
        ("Transaction CSV", dataset_path),
        ("Test evaluation", test_report_path),
        ("Model benchmark", benchmark_path),
        ("Threshold report", threshold_path),
        ("Cost sensitivity report", cost_path),
        ("Feature ablation", ablation_path),
        ("Origin feature ablation", origin_ablation_path),
        ("Feature importance", importance_path),
        ("Split report", split_path),
        ("Data quality report", quality_path),
    ]

    source_df = pd.DataFrame([
        {
            "Artifact": label,
            "Status": "FOUND" if path else "MISSING",
            "Path": str(path.relative_to(ROOT)) if path else "Not found",
        }
        for label, path in source_paths
    ])

    st.dataframe(
        source_df,
        use_container_width=True,
        hide_index=True,
    )

    if quality_report:
        st.markdown("### Data quality report")
        st.json(quality_report, expanded=True)

    if stats:
        st.markdown("### Dataset scan")
        st.json(
            {
                "rows_scanned": stats["transactions"],
                "fraud_count": stats["fraud"],
                "fraud_rate": stats["fraud_rate"],
                "sample_size": stats["sample_size"],
                "columns": stats["columns"],
            },
            expanded=True,
        )

    st.markdown(
        """
        <div class="note">
        <b>How the dashboard works:</b><br>
        • Headline transaction and fraud counts are computed from the full CSV.<br>
        • Interactive transaction charts use a bounded sample for responsiveness.<br>
        • Model and cost metrics come from saved report files.<br>
        • Missing files are reported rather than replaced with fake results.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "FINTELLIGENCE · FinTech Fraud Model Benchmark · "
    "Synthetic research dataset · Not a production decision system"
)
