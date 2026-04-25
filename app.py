import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Sistem Prediksi Kelulusan Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS - LIGHT MODE THEME
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #F8FAFC;
}

/* ===== HEADER ===== */
.dev-header {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    padding: 1.75rem 2rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 2rem;
    box-shadow: 0 10px 40px rgba(37, 99, 235, 0.2);
}

.dev-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.8rem;
    margin: 0;
    letter-spacing: -0.5px;
}

.dev-header p {
    font-family: 'Inter', sans-serif;
    font-weight: 400;
    opacity: 0.95;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
}

/* ===== CARDS ===== */
.card {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    border: 1px solid #E2E8F0;
    margin-bottom: 1.5rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
}

.card-title {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.25rem;
    color: #1E293B;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ===== BUTTONS ===== */
.stButton>button {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
    background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
}

/* ===== BADGES ===== */
.badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0.75rem 1.5rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 1.1rem;
    font-family: 'Inter', sans-serif;
    gap: 0.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.badge-success {
    background: linear-gradient(135deg, #10B981 0%, #059669 100%);
    color: white;
}

.badge-danger {
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white;
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    border: 1px solid #E2E8F0;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2563EB;
    font-family: 'Inter', sans-serif;
}

.metric-label {
    font-size: 0.9rem;
    color: #64748B;
    margin-top: 0.25rem;
    font-family: 'Inter', sans-serif;
}

/* ===== SLIDER STYLING ===== */
.stSlider [data-baseweb="slider"] div div {
    background: #2563EB !important;
}

.stSlider [data-testid="stThumbValue"] {
    color: #2563EB !important;
    font-weight: 600 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px 8px 0 0 !important;
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    color: #1E293B;
    margin-bottom: 1.5rem;
    letter-spacing: -0.5px;
}

/* ===== DATAFRAME ===== */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ===== SIDEBAR ===== */
.css-1d391kg {
    background-color: #FFFFFF !important;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    padding: 2rem;
    color: #94A3B8;
    font-size: 0.85rem;
    border-top: 1px solid #E2E8F0;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎓</div>
    <div style="font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1.1rem; color: #1E293B;">Prediksi Kelulusan</div>
    <div style="font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #64748B; margin-top: 0.25rem;">Decision Tree Analytics</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Dashboard", "📊 Dataset", "🌳 Model & Visualisasi", "🔮 Prediksi"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="padding: 1rem; background: #F1F5F9; border-radius: 12px; margin-top: 1rem;">
    <div style="font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #64748B; margin-bottom: 0.5rem;">👩‍💻 PENGEMBANG</div>
    <div style="font-family: 'Inter', sans-serif; font-weight: 600; color: #1E293B; font-size: 0.9rem;">Een Erna Wati</div>
    <div style="font-family: 'Inter', sans-serif; color: #2563EB; font-size: 0.8rem;">NIM: 23050915</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# DATASET
# ============================================================
@st.cache_data
def load_data():
    data = {
        'Nama Mahasiswa': [
            'Een Erna Wati', 'Nurul Fazri', 'Fahmiyansah', 'Faisal Rahman',
            'Serlinaa', 'Dita Amelia', 'Junita', 'Zajuli',
            'Ika Kurniasih', 'Lusi Kawati'
        ],
        'IPK': [3.75, 2.10, 3.40, 1.85, 3.90, 2.45, 3.60, 1.60, 3.25, 2.80],
        'Kehadiran (%)': [92, 65, 88, 55, 96, 70, 85, 50, 82, 75],
        'Nilai Tugas': [88, 60, 82, 50, 95, 65, 85, 45, 78, 72],
        'Nilai Ujian': [85, 55, 80, 48, 92, 62, 88, 42, 75, 70],
        'Status Kelulusan': ['Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus',
                             'Lulus', 'Tidak Lulus', 'Lulus', 'Tidak Lulus',
                             'Lulus', 'Lulus']
    }
    df = pd.DataFrame(data)
    df['Status Biner'] = df['Status Kelulusan'].map({'Lulus': 1, 'Tidak Lulus': 0})
    return df

df = load_data()

# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource
def train_model():
    X = df[['IPK', 'Kehadiran (%)', 'Nilai Tugas', 'Nilai Ujian']]
    y = df['Status Biner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Tidak Lulus', 'Lulus'], output_dict=True)
    return model, accuracy, report, X.columns.tolist()

model, accuracy, report, feature_names = train_model()

# ============================================================
# HEADER COMPONENT
# ============================================================
def render_header():
    st.markdown("""
    <div class="dev-header">
        <h1>🎓 Sistem Prediksi Kelulusan Mahasiswa</h1>
        <p>Pengembang: <strong>Een Erna Wati</strong> (NIM: 23050915) &nbsp;|&nbsp; Metode: Decision Tree Classifier</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "🏠 Dashboard":
    render_header()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Mahasiswa</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        lulus_count = len(df[df['Status Kelulusan'] == 'Lulus'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #10B981;">{lulus_count}</div>
            <div class="metric-label">Lulus</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        tidak_lulus_count = len(df[df['Status Kelulusan'] == 'Tidak Lulus'])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #EF4444;">{tidak_lulus_count}</div>
            <div class="metric-label">Tidak Lulus</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #8B5CF6;">{accuracy*100:.1f}%</div>
            <div class="metric-label">Akurasi Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Distribusi Status Kelulusan</div>
        </div>
        """, unsafe_allow_html=True)
        status_counts = df['Status Kelulusan'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Jumlah']
        fig = px.pie(status_counts, values='Jumlah', names='Status',
                     color='Status',
                     color_discrete_map={'Lulus': '#10B981', 'Tidak Lulus': '#EF4444'},
                     hole=0.5)
        fig.update_layout(
            font=dict(family="Inter, sans-serif"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("""
        <div class="card">
            <div class="card-title">📋 Ringkasan Model</div>
            <div style="font-family: 'Inter', sans-serif; color: #475569; line-height: 1.8;">
                <p><strong>Algoritma:</strong> Decision Tree Classifier</p>
                <p><strong>Max Depth:</strong> 4</p>
                <p><strong>Split:</strong> 70% Train / 30% Test</p>
                <p><strong>Akurasi:</strong> {:.1f}%</p>
                <p><strong>Fitur:</strong> IPK, Kehadiran, Nilai Tugas, Nilai Ujian</p>
            </div>
        </div>
        """.format(accuracy*100), unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Dikembangkan oleh Een Erna Wati (NIM: 23050915) &nbsp;|&nbsp; Sistem Prediksi Kelulusan Mahasiswa &copy; 2025
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: DATASET
# ============================================================
elif page == "📊 Dataset":
    render_header()

    st.markdown("""
    <div class="card">
        <div class="card-title">📋 Data Mahasiswa</div>
        <p style="color: #64748B; font-family: 'Inter', sans-serif; margin-bottom: 1.5rem;">
            Dataset berikut berisi informasi akademik 10 mahasiswa yang digunakan untuk melatih model Decision Tree.
        </p>
    </div>
    """, unsafe_allow_html=True)

    styled_df = df[['Nama Mahasiswa', 'IPK', 'Kehadiran (%)', 'Nilai Tugas', 'Nilai Ujian', 'Status Kelulusan']].copy()
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Nama Mahasiswa': st.column_config.TextColumn('Nama Mahasiswa', width='large'),
            'IPK': st.column_config.NumberColumn('IPK', format='%.2f'),
            'Kehadiran (%)': st.column_config.NumberColumn('Kehadiran (%)', format='%d%%'),
            'Nilai Tugas': st.column_config.NumberColumn('Nilai Tugas'),
            'Nilai Ujian': st.column_config.NumberColumn('Nilai Ujian'),
            'Status Kelulusan': st.column_config.TextColumn('Status Kelulusan')
        }
    )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">📈 Statistik Deskriptif</div>
        </div>
        """, unsafe_allow_html=True)
        desc = df[['IPK', 'Kehadiran (%)', 'Nilai Tugas', 'Nilai Ujian']].describe().round(2)
        st.dataframe(desc, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Perbandingan Status</div>
        </div>
        """, unsafe_allow_html=True)
        fig_bar = px.bar(
            df['Status Kelulusan'].value_counts().reset_index(),
            x='Status Kelulusan',
            y='count',
            color='Status Kelulusan',
            color_discrete_map={'Lulus': '#10B981', 'Tidak Lulus': '#EF4444'},
            text='count'
        )
        fig_bar.update_layout(
            font=dict(family="Inter, sans-serif"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("""
    <div class="footer">
        Dikembangkan oleh Een Erna Wati (NIM: 23050915) &nbsp;|&nbsp; Sistem Prediksi Kelulusan Mahasiswa &copy; 2025
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: MODEL & VISUALISASI
# ============================================================
elif page == "🌳 Model & Visualisasi":
    render_header()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🎯 Akurasi Model</div>
            <div style="text-align: center; padding: 2rem 0;">
                <div style="font-size: 4rem; font-weight: 700; color: #2563EB; font-family: 'Inter', sans-serif;">{:.1f}%</div>
                <div style="color: #64748B; font-family: 'Inter', sans-serif; margin-top: 0.5rem;">Akurasi Decision Tree</div>
            </div>
        </div>
        """.format(accuracy*100), unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">📊 Classification Report</div>
        </div>
        """, unsafe_allow_html=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        st.dataframe(report_df, use_container_width=True)

    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🔥 Feature Importance</div>
        </div>
        """, unsafe_allow_html=True)
        importance_df = pd.DataFrame({
            'Fitur': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(
            importance_df,
            x='Importance',
            y='Fitur',
            orientation='h',
            color='Importance',
            color_continuous_scale=['#93C5FD', '#2563EB', '#1E40AF']
        )
        fig_imp.update_layout(
            font=dict(family="Inter, sans-serif"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col2:
