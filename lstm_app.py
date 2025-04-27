import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components

# ✅ 페이지 설정
st.set_page_config(page_title="CPI 예측 대시보드", layout="wide")
st.title("📈 CPI 예측 모델 대시보드")

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ 모델과 스케일러 불러오기
@st.cache_resource
def load_model():
    return joblib.load("cpi_model.pkl")

@st.cache_resource
def load_scaler():
    return joblib.load("scaler.pkl")

model = load_model()
scaler = load_scaler()

# ✅ 컬럼명 처리
def rename_columns_for_profiling(df):
    mapping = {'CPI': 'CPI', '금리': 'Interest Rate', '환율': 'Exchange Rate', '날짜': 'Date'}
    return df.rename(columns=mapping), mapping

def revert_column_names(df, mapping):
    reverse_mapping = {v: k for k, v in mapping.items()}
    return df.rename(columns=reverse_mapping)

# ✅ 상관관계 히트맵
def plot_corr_heatmap(df):
    corr = df[["CPI", "금리", "환율"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("📊 변수 간 상관관계")
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    st.image(buf)

# ✅ 사이드바 CSV 업로드
st.sidebar.header("📂 CSV 업로드")
uploaded_file = st.sidebar.file_uploader("CPI 예측용 CSV 파일을 업로드하세요", type="csv")

# ✅ 사용자 직접 입력을 통한 예측
with st.sidebar.expander("🧮 직접 입력으로 CPI 예측하기"):
    cpi_input = st.number_input("CPI (이전 월 등)", min_value=0.0, format="%.2f")
    rate_input = st.number_input("금리", min_value=0.0, format="%.2f")
    ex_input = st.number_input("환율", min_value=0.0, format="%.2f")

    if st.button("📌 입력값으로 예측하기"):
        try:
            # 최근 12개월 더미 입력을 구성 (입력값을 마지막에 붙임)
            dummy = np.zeros((11, 3))  # 임의로 앞 11개는 0으로 채움
            user_input = np.array([[cpi_input, rate_input, ex_input]])
            full_input = np.vstack([dummy, user_input])

            scaled_input = scaler.transform(full_input)
            X_input = scaled_input.reshape(1, 12, 3)

            y_pred = model.predict(X_input).flatten()[0]
            # 역정규화: CPI 값만 추출
            inv = scaler.inverse_transform(
                np.concatenate([[y_pred], [0], [0]]).reshape(1, 3)
            )[0, 0]

            st.success(f"✅ 예측된 CPI: **{inv:.2f}**")
        except Exception as e:
            st.error(f"예측 중 오류 발생: {e}")

# ✅ 파일 업로드가 있을 때만 수행
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if '날짜' not in df.columns:
        st.error("❌ '날짜' 컬럼이 존재해야 합니다.")
        st.stop()

    df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')
    st.subheader("✅ 업로드된 원본 데이터")
    st.dataframe(df.head())

    renamed_df, col_map = rename_columns_for_profiling(df)

    st.subheader("📊 데이터 프로파일링 리포트")
    profile = ProfileReport(renamed_df, title="CPI 데이터 분석", explorative=True)
    with st.spinner("데이터 프로파일링 리포트를 생성 중입니다..."):
        profile_html = profile.to_file("profile_report.html")
        with open("profile_report.html", "r", encoding='utf-8') as f:
           html = f.read()
        components.html(html, height=1000, scrolling=True)

    df = revert_column_names(renamed_df, col_map)

    st.subheader("🧊 CPI, 금리, 환율 간 상관관계 히트맵")
    plot_corr_heatmap(df)

    feature_cols = ['CPI', '금리', '환율']

    if all(col in df.columns for col in feature_cols):
        for col in feature_cols:
            df[col] = df[col].astype(str).str.replace(",", "").astype(float)

        X = df[feature_cols].values
        scaled_X = scaler.transform(X)

        look_back = 12
        X_lstm = []
        for i in range(len(scaled_X) - look_back):
            X_lstm.append(scaled_X[i:i+look_back])
        X_lstm = np.array(X_lstm)

        if len(X_lstm) == 0:
            st.error("❌ 데이터가 너무 적어 LSTM 시퀀스를 만들 수 없습니다.")
            st.stop()

        y_pred = model.predict(X_lstm).flatten()

        df = df.iloc[look_back:].copy()
        df['CPI_pred'] = scaler.inverse_transform(
            np.concatenate([y_pred.reshape(-1, 1), np.zeros((len(y_pred), 2))], axis=1)
        )[:, 0]

        st.subheader("📈 실제 vs 예측 CPI 시계열")
        st.line_chart(df.set_index("날짜")[['CPI', 'CPI_pred']])

        st.subheader("📌 예측 성능 지표")
        mae = mean_absolute_error(df['CPI'], df['CPI_pred'])
        rmse = np.sqrt(mean_squared_error(df['CPI'], df['CPI_pred']))
        r2 = r2_score(df['CPI'], df['CPI_pred'])

        col1, col2, col3 = st.columns(3)
        col1.metric("📉 MAE", f"{mae:.4f}")
        col2.metric("📊 RMSE", f"{rmse:.4f}")
        col3.metric("📈 R² Score", f"{r2:.4f}")
    else:
        st.error("❌ 'CPI', '금리', '환율' 컬럼이 모두 존재해야 합니다.")
else:
    st.info("📂 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")
