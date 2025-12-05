import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------------
# 1. 페이지 설정
# -----------------------------
st.set_page_config(page_title="마르코프 체인 IDS", layout="wide")
st.title("🔒 마르코프 체인 기반 네트워크 침입 탐지 시스템")
st.write("""
이 앱은 정상 네트워크 트래픽 데이터를 기반으로 **연속 이벤트 전이 패턴**을 학습하고,
새로운 연결 이벤트에서 이상 징후를 탐지합니다.
""")

# -----------------------------
# 2. 사이드바
# -----------------------------
st.sidebar.header("설정")
scenario = st.sidebar.selectbox(
    "분석 시나리오 선택",
    options=["정상 트래픽", "DoS 공격", "Probe 공격", "혼합 시나리오", "랜덤 시나리오"]
)

seq_len = st.sidebar.slider("연속 이벤트 길이", min_value=2, max_value=20, value=5, step=1)
seq_count = st.sidebar.slider("연속 이벤트 묶음 수", min_value=10, max_value=500, value=50, step=10)
threshold = st.sidebar.slider("이상 탐지 민감도", min_value=0.0001, max_value=0.1, value=0.01, step=0.001)

with st.sidebar.expander("💡 탐지 도움말"):
    st.write("""
    - 빨간 점: 이상 이벤트 탐지  
    - 파란 점: 정상 이벤트  
    - 평균 전이 확률이 임계값보다 낮으면 이상 이벤트로 표시됩니다.
    """)

# -----------------------------
# 3. 데이터 로드
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
        'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty'
    ]
    df = pd.read_csv(file_path, names=column_names, index_col=False)
    df['label'] = df['label'].astype(str).str.strip().str.replace(r'\.', '', regex=True)
    return df

data_train = load_data("KDDTrain+.txt")
st.success("✅ 데이터 로드 완료")

# -----------------------------
# 4. 마르코프 전이행렬 생성
# -----------------------------
def create_transition_matrix(df_normal):
    # 1) 정상 데이터에서 등장하는 flag 상태들을 수집하고 정렬
    states = sorted(df_normal['flag'].unique())
    
    # 2) 상태를 행렬 인덱스로 매핑하기 위한 딕셔너리 생성
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    # 3) (상태 수 x 상태 수) 크기의 전이 카운트 행렬 생성
    num_states = len(states)
    counts = np.zeros((num_states, num_states))
    
    # 4) flag 값 시퀀스를 리스트로 추출
    flags = df_normal['flag'].tolist()
    
    # 5) 연속된 flag 전이 횟수를 카운트
    for i in range(len(flags) - 1):
        current_state = state_to_idx[flags[i]]
        next_state = state_to_idx[flags[i + 1]]
        counts[current_state, next_state] += 1
    
    # 6) 각 상태별 총 발생 수로 나누어 확률 계산 (정규화)
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(counts, row_sums, where=row_sums != 0)
    
    # 7) 0으로 나누는 경우 발생 시 확률값을 0.0001로 대체하여 안정성 확보
    transition_matrix = np.nan_to_num(transition_matrix, nan=0.0001)
    
    # 8) 결과를 DataFrame 형태로 반환
    return pd.DataFrame(transition_matrix, index=states, columns=states)

# 정상 데이터만 추출하여 모델 학습
df_normal = data_train[data_train['label'] == "normal"]
transition_model = create_transition_matrix(df_normal)


st.subheader("📊 학습된 정상 트래픽 전이행렬 (일부)")
st.dataframe(transition_model.round(3))
st.caption("행: 현재 상태, 열: 다음 상태, 값: 발생 확률")

# -----------------------------
# 5. 탐지 시뮬레이션 (시나리오 반영)
# -----------------------------
np.random.seed(42)
states = list(transition_model.index)
sim_sequences = []
avg_probs = []

for _ in range(seq_count):
    if scenario == "정상 트래픽":
        seq = np.random.choice(states, seq_len)

    elif scenario == "DoS 공격":
        # 약한 상태로 몰리는 패턴(비정상 flag)
        bad_state = states[-1]
        seq = np.random.choice(states + [bad_state], seq_len,
                               p=[0.7/len(states)]*len(states) + [0.3])

    elif scenario == "Probe 공격":
        # 탐색 시도 → 불규칙한 잦은 이동
        seq = np.random.choice(states, seq_len,
                               p=[1/len(states)]*len(states))
        np.random.shuffle(seq)

    elif scenario == "혼합 시나리오":
        seq = np.random.choice(states, seq_len)
        if np.random.rand() < 0.3:
            seq = seq[::-1]  # 뒤집어 이상 패턴 생성

    elif scenario == "랜덤 시나리오":
        seq = np.random.choice(states, seq_len)

    sim_sequences.append(seq)

    # 평균 전이 확률
    prob_list = [transition_model.loc[seq[i], seq[i+1]] for i in range(len(seq)-1)]
    avg_prob = np.mean(prob_list)
    avg_probs.append(avg_prob)

anomaly_flags = [p < threshold for p in avg_probs]

# -----------------------------
# 6. Plotly 그래프
# -----------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(1, seq_count+1)), y=avg_probs,
    mode="lines+markers", name="평균 전이 확률"
))
fig.add_trace(go.Scatter(
    x=[i+1 for i, flag in enumerate(anomaly_flags) if flag],
    y=[avg_probs[i] for i, flag in enumerate(anomaly_flags) if flag],
    mode="markers", marker=dict(color="red", size=10), name="이상 이벤트"
))
fig.update_layout(title="연속 이벤트 평균 전이 확률", xaxis_title="연속 이벤트 묶음 번호",
                  yaxis_title="평균 전이 확률")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 7. 탐지 결과 테이블
# -----------------------------
st.subheader("🔍 탐지 결과 요약")
df_plot = pd.DataFrame({
    "연속 이벤트 묶음 번호": range(1, seq_count+1),
    "평균 전이 확률": avg_probs,
    "이상 탐지 여부": ["이상" if flag else "정상" for flag in anomaly_flags]
})
st.dataframe(df_plot.head(20))
