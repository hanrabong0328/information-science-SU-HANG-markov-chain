import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="마르코프 체인 IDS", layout="wide")
st.title("🔒 마르코프 체인 기반 네트워크 침입 탐지 시스템")
st.write("""
이 앱은 정상 네트워크 연결 데이터를 기반으로 **연속 이벤트 전이 패턴**을 학습하고,
새로운 연결 이벤트에서 이상 징후를 탐지합니다.
""")

# -----------------------------
# 1. 사이드바 - 사용자 입력
# -----------------------------
st.sidebar.header("1️⃣ 탐지 설정")
scenario = st.sidebar.selectbox(
    "분석 시나리오 선택",
    options=["정상 트래픽", "DoS 공격", "Probe 공격", "혼합 시나리오", "랜덤 시나리오"],
    help="분석할 트래픽 유형을 선택하세요."
)
seq_len = st.sidebar.slider(
    "연속 이벤트 길이", min_value=2, max_value=20, value=5, step=1,
    help="한 번에 관찰할 연속 이벤트 수 설정"
)
seq_count = st.sidebar.slider(
    "연속 이벤트 묶음 수", min_value=10, max_value=500, value=50, step=10,
    help="한 번에 분석할 이벤트 묶음 수 설정"
)
threshold = st.sidebar.slider(
    "이상 탐지 민감도",
    min_value=0.0001, max_value=0.1, value=0.01, step=0.001,
    help="이 확률보다 낮으면 이상 이벤트로 탐지"
)
st.sidebar.info("슬라이더 조정 시 결과 그래프와 탐지 결과가 실시간 업데이트 됩니다.")

# -----------------------------
# 2. 데이터 로드
# -----------------------------
@st.cache_data
def load_data(file_path):
    column_names = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes',
        'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
        'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
        'num_shells','num_access_files','num_outbound_cmds','is_host_login',
        'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
        'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
        'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
        'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
        'dst_host_rerror_rate','dst_host_srv_rerror_rate',
        'label','difficulty'
    ]
    df = pd.read_csv(file_path, names=column_names, index_col=False)
    df['label'] = df['label'].astype(str).str.strip().str.replace(r'\.', '', regex=True)
    return df

data_train = load_data("KDDTrain+.txt")
st.success("✅ 데이터 로드 완료")

# -----------------------------
# 3. 전이행렬 생성 함수
# -----------------------------
def create_transition_matrix(df_normal):
    states = sorted(df_normal['flag'].unique())
    state_to_idx = {state:i for i,state in enumerate(states)}
    num_states = len(states)
    counts = np.zeros((num_states,num_states))
    flags = df_normal['flag'].tolist()
    for i in range(len(flags)-1):
        counts[state_to_idx[flags[i]], state_to_idx[flags[i+1]]] +=1
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(counts, row_sums, where=row_sums != 0)
    transition_matrix = np.nan_to_num(transition_matrix, nan=0.0001)
    return pd.DataFrame(transition_matrix, index=states, columns=states)

df_normal = data_train[data_train['label']=="normal"]
transition_model = create_transition_matrix(df_normal)

# -----------------------------
# 4. 시뮬레이션 탐지 (예시)
# -----------------------------
np.random.seed(42)
states = list(transition_model.index)
sim_sequences = []
avg_probs = []

for _ in range(seq_count):
    seq = np.random.choice(states, seq_len)
    sim_sequences.append(seq)
    prob = 1.0
    for i in range(len(seq)-1):
        prob *= transition_model.loc[seq[i], seq[i+1]]
    avg_probs.append(prob)

anomaly_flags = [p<threshold for p in avg_probs]

# -----------------------------
# 5. Plotly 그래프
# -----------------------------
df_plot = pd.DataFrame({
    "연속 이벤트 묶음 번호": range(1, seq_count+1),
    "평균 전이 확률": avg_probs,
    "이상 탐지 여부":["이상" if flag else "정상" for flag in anomaly_flags]
})
fig = px.scatter(
    df_plot, x="연속 이벤트 묶음 번호", y="평균 전이 확률",
    color="이상 탐지 여부", color_discrete_map={"정상":"blue","이상":"red"},
    title="연속 이벤트 평균 전이 확률 (빨간 점 = 이상)"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6. 탐지 결과 테이블
# -----------------------------
st.subheader("🔍 탐지 결과 요약")
st.dataframe(df_plot.head(20))

# -----------------------------
# 7. 성능 지표 (예시)
# -----------------------------
st.subheader("📊 성능 지표 예시")
st.write("""
- **TP**: 올바르게 탐지한 공격  
- **FP**: 잘못 탐지한 정상  
- **TN**: 정상으로 정확히 판단  
- **FN**: 탐지 못한 공격  
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
""")
st.info("※ 실제 테스트 데이터가 있으면 TP, FP, TN, FN 값 계산 가능")
