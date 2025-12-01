# streamlit_app.py
# 한글 주석 포함 — 시나리오 선택 + Threshold 실시간 반영 + 시각화 + 성능 지표
import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go

# =======================================
# 설정: GitHub Raw URL (사용자 리포지토리)
# =======================================
GITHUB_RAW_URL = "https://raw.githubusercontent.com/hanrabong0328/information-science-SU-HANG-markov-chain/main/KDDTrain+.txt"

# =======================================
# 데이터 로드 함수 (한번 캐시)
# =======================================
@st.cache_data(show_spinner=True)
def load_kdd(url=GITHUB_RAW_URL):
    """
    NSL-KDD 데이터 로드: 공백(또는 자동) 구분으로 읽습니다.
    컬럼 이름을 미리 정의하여 나중에 편하게 사용합니다.
    """
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
        'label', 'difficulty'
    ]
    # sep=None + engine='python'로 공백/콤마 자동 처리
    df = pd.read_csv(url, names=column_names, sep=None, engine='python', index_col=False)
    # 라벨 문자열 정리 (끝에 점(.) 붙는 경우 제거)
    df['label'] = df['label'].astype(str).str.strip().str.replace(r'\.', '', regex=True)
    return df

# =======================================
# 마르코프 전이행렬 생성 (정상 데이터만 사용)
# =======================================
def create_transition_matrix_from_flag(df_normal, smoothing=1e-6):
    """
    flag 열을 상태로 보고, 정상 데이터에서 전이 행렬을 계산합니다.
    smoothing: unseen transition에 작은 확률을 부여하기 위한 값
    반환: pandas.DataFrame (index/columns = 상태)
    """
    states = sorted(df_normal['flag'].unique())
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)))
    flags = df_normal['flag'].tolist()
    for i in range(len(flags) - 1):
        a = idx[flags[i]]
        b = idx[flags[i+1]]
        counts[a, b] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    # 확률 계산 (0으로 나누는 상황 방지)
    probs = np.divide(counts, row_sums, where=row_sums != 0)
    # row_sums==0인 행(전혀 관측되지 않음)에 작은 확률 부여
    for i in range(len(states)):
        if row_sums[i, 0] == 0:
            probs[i, :] = smoothing
    return pd.DataFrame(probs, index=states, columns=states)

# =======================================
# 시나리오별 테스트 시퀀스 생성
# =======================================
def generate_scenario_sequences(df, scenario='Normal', seq_len=10, n_sequences=200, mix_ratio=0.5, random_state=42):
    """
    시나리오에 따라 테스트용 시퀀스(플래그 시퀀스)를 생성합니다.
    - Normal: 정상 라벨로 구성된 시퀀스
    - DoS: DoS 계열 라벨(예: neptune, smurf)을 우선적으로 사용
    - Probe: Probe 계열 (satan, ipsweep, nmap 등)
    - Mixed: 정상 + 공격 섞음 (mix_ratio로 공격 비율 조정)
    - Random: 전체에서 랜덤 샘플링
    반환: list of dicts: {'flags': [...], 'true_label': 0 or 1}
    (true_label: 0=normal sequence, 1=attack sequence) — 라벨링은 시퀀스 내 행의 라벨 비중으로 결정
    """
    np.random.seed(random_state)
    sequences = []

    # 공격 그룹 정의 (NSL-KDD 라벨 예시)
    dos_labels = ['neptune', 'smurf', 'back', 'teardrop']  # 대표적인 DoS 라벨
    probe_labels = ['satan', 'ipsweep', 'nmap', 'portsweep']

    # 레이블별 인덱 추출
    mask_normal = df['label'].str.contains('normal', case=False, na=False)
    df_normal = df[mask_normal].reset_index(drop=True)
    df_dos = df[df['label'].str.lower().isin(dos_labels)].reset_index(drop=True)
    df_probe = df[df['label'].str.lower().isin(probe_labels)].reset_index(drop=True)
    df_all = df.reset_index(drop=True)

    def sample_flags_from_df(df_src, k):
        # 연속 윈도우로 뽑기: 시작 인덱스 랜덤 선택 후 seq_len 길이로 슬라이싱(있으면)
        if len(df_src) == 0:
            # 해당 라벨이 없으면 랜덤 전부 사용
            idxs = np.random.randint(0, len(df_all), size=k)
            return [list(df_all.loc[i:i+seq_len-1, 'flag'].values) if i+seq_len-1 < len(df_all) else list(df_all.loc[i:len(df_all)-1, 'flag'].values) + list(df_all.loc[0: (seq_len - (len(df_all)-i)), 'flag'].values) for i in idxs]
        else:
            idxs = np.random.randint(0, max(1, len(df_src)-seq_len), size=k)
            seqs = []
            for i in idxs:
                seq = list(df_src.loc[i:i+seq_len-1, 'flag'].values)
                # 길이 부족하면 랜덤 패딩
                if len(seq) < seq_len:
                    extra = list(df_all.sample(seq_len - len(seq))['flag'].values)
                    seq = seq + extra
                seqs.append(seq)
            return seqs

    if scenario == 'Normal':
        seqs = sample_flags_from_df(df_normal, n_sequences)
        for s in seqs:
            sequences.append({'flags': s, 'true_label': 0})
    elif scenario == 'DoS':
        seqs = sample_flags_from_df(df_dos, n_sequences)
        for s in seqs:
            sequences.append({'flags': s, 'true_label': 1})
    elif scenario == 'Probe':
        seqs = sample_flags_from_df(df_probe, n_sequences)
        for s in seqs:
            sequences.append({'flags': s, 'true_label': 1})
    elif scenario == 'Mixed':
        # mix_ratio 비율만큼 공격(DoS+Probe 혼합), 나머지 정상
        n_attack = int(n_sequences * mix_ratio)
        n_normal = n_sequences - n_attack
        seqs_normal = sample_flags_from_df(df_normal, n_normal)
        seqs_dos = sample_flags_from_df(df_dos, n_attack//2 + n_attack%2)
        seqs_probe = sample_flags_from_df(df_probe, n_attack//2)
        for s in seqs_normal:
            sequences.append({'flags': s, 'true_label': 0})
        for s in seqs_dos:
            sequences.append({'flags': s, 'true_label': 1})
        for s in seqs_probe:
            sequences.append({'flags': s, 'true_label': 1})
        # 셔플
        np.random.shuffle(sequences)
        sequences = sequences[:n_sequences]
    elif scenario == 'Random':
        # 전체에서 랜덤으로 시퀀스 생성 (레이블은 원래 비율대로)
        seqs = []
        for _ in range(n_sequences):
            start = np.random.randint(0, max(1, len(df_all)-seq_len))
            seq = list(df_all.loc[start:start+seq_len-1, 'flag'].values)
            if len(seq) < seq_len:
                extra = list(df_all.sample(seq_len - len(seq))['flag'].values)
                seq = seq + extra
            # true label: sequence 내 공격 비율 > 0.5 => 1
            lbls = list(df_all.loc[start:start+seq_len-1, 'label'].values)
            if len(lbls) < seq_len:
                lbls = list(df_all.loc[start:len(df_all)-1, 'label'].values) + list(df_all.sample(seq_len - len(lbls))['label'].values)
            true_label = 1 if sum([0 if 'normal' in str(x).lower() else 1 for x in lbls]) / seq_len > 0.5 else 0
            sequences.append({'flags': seq, 'true_label': true_label})
    else:
        raise ValueError("Unknown scenario")
    return sequences

# =======================================
# 시퀀스 점수 계산 함수
# =======================================
def score_sequence_mean_prob(seq_flags, trans_df, eps=1e-12):
    """
    주어진 flag 시퀀스(길이 L)에 대해 각 전이의 확률을 전이행렬에서 읽어
    평균 전이확률(mean transition probability)을 반환.
    - unseen transition은 eps 처리
    """
    probs = []
    for i in range(len(seq_flags) - 1):
        a = seq_flags[i]; b = seq_flags[i+1]
        if a in trans_df.index and b in trans_df.columns:
            p = trans_df.loc[a, b]
            # numeric 안전성
            if pd.isna(p) or p <= 0:
                p = eps
        else:
            p = eps
        probs.append(p)
    if len(probs) == 0:
        return eps
    return float(np.mean(probs))

# =======================================
# Streamlit UI — 화면 구성
# =======================================
st.set_page_config(layout="wide", page_title="Markov-IDS Interactive")
st.title("📡 Markov Chain 기반 이상징후 탐지 데모 (인터랙티브)")
st.markdown("시나리오 선택 → 임계값 조정 → 탐지 결과와 시각화가 실시간으로 변합니다.")

# 사이드바: 시나리오 선택
st.sidebar.header("1) 시나리오 선택")
scenario = st.sidebar.selectbox("테스트 트래픽 시나리오 선택",
                                ['Normal', 'DoS', 'Probe', 'Mixed', 'Random'])

# 시퀀스 길이 및 개수
st.sidebar.header("2) 시퀀스 설정")
seq_len = st.sidebar.slider("시퀀스 길이 (연속 이벤트 수)", min_value=3, max_value=50, value=10, step=1)
n_seq = st.sidebar.slider("생성할 시퀀스 개수", min_value=50, max_value=1000, value=200, step=50)

# 임계값(Probability 기반)
st.sidebar.header("3) 임계값 설정 (Probability 기준)")
threshold = st.sidebar.slider("탐지 임계값 (mean transition probability)",
                              min_value=0.0, max_value=0.1, value=0.01, step=0.001)
st.sidebar.caption("시퀀스의 mean transition probability < threshold 이면 '이상'으로 분류합니다.")

# Load data (캐시)
with st.spinner("데이터 로드 및 모델 학습 중..."):
    df_all = load_kdd()
    # 정상 데이터만 골라 학습
    df_train_normal = df_all[df_all['label'].str.contains('normal', case=False, na=False)].reset_index(drop=True)
    trans_df = create_transition_matrix_from_flag(df_train_normal)

# 좌측: 설정 요약 / 우측: 그래프
left_col, right_col = st.columns([1,2])

with left_col:
    st.subheader("설정 요약")
    st.write(f"- 시나리오: **{scenario}**")
    st.write(f"- 시퀀스 길이: **{seq_len}**")
    st.write(f"- 시퀀스 수: **{n_seq}**")
    st.write(f"- 임계값(Mean prob): **{threshold:.4f}**")
    st.write("---")
    st.subheader("전이행렬 (상위 일부)")
    st.dataframe(trans_df.head(8).round(4))

with right_col:
    st.subheader("탐지 결과 시각화 및 지표")

    # 시나리오에 따라 시퀀스 생성
    sequences = generate_scenario_sequences(df_all, scenario=scenario, seq_len=seq_len, n_sequences=n_seq)

    # 각 시퀀스 점수 계산
    scores = []
    true_labels = []
    for sdict in sequences:
        seq = sdict['flags']
        score = score_sequence_mean_prob(seq, trans_df)
        scores.append(score)
        true_labels.append(sdict['true_label'])

    # anomaly 결정 (threshold 기준)
    pred_anoms = [1 if sc < threshold else 0 for sc in scores]

    # Plotly: 점수 시계열 (x=index, y=score), 이상은 빨간 점
    df_plot = pd.DataFrame({
        'idx': list(range(len(scores))),
        'score': scores,
        'pred_anom': pred_anoms,
        'true_label': true_labels
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['idx'], y=df_plot['score'],
                             mode='lines+markers', name='Mean transition prob',
                             marker=dict(size=6)))
    # 이상 탐지 포인트 (pred)
    anom_df = df_plot[df_plot['pred_anom'] == 1]
    if len(anom_df) > 0:
        fig.add_trace(go.Scatter(x=anom_df['idx'], y=anom_df['score'],
                                 mode='markers', name='Detected Anomaly (pred)',
                                 marker=dict(color='red', size=8, symbol='x')))
    # 임계값 라인
    fig.add_hline(y=threshold, line_dash="dash", annotation_text="Threshold", annotation_position="bottom right")

    fig.update_layout(height=450, xaxis_title="Sequence index", yaxis_title="Mean transition probability")
    st.plotly_chart(fig, use_container_width=True)

    # 성능 지표: true_labels이 있으면 계산
    if any([x is not None for x in true_labels]):
        y_true = np.array(true_labels)
        y_pred = np.array(pred_anoms)
        # confusion matrix (0: normal, 1: attack)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        except:
            # 라벨 구성에 따라 에러 날 수 있으니 안전 처리
            cm = confusion_matrix(y_true, y_pred, labels=np.unique(np.concatenate([y_true, y_pred])))
            # 간단한 fallback
            tn = fp = fn = tp = 0
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        st.write("### 성능 지표")
        st.write({
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Precision": float(np.round(precision, 4)), "Recall": float(np.round(recall, 4))
        })
    else:
        st.info("true labels이 없어서 성능 지표를 계산할 수 없습니다.")

    # 상위 이상(예: score 낮은 순) 샘플 표 보여주기
    st.write("### 탐지된 상위 이상(혹은 낮은 score) 샘플")
    df_sample = df_plot.sort_values('score').head(10).reset_index(drop=True)
    st.dataframe(df_sample)

# =======================================
# 사용 팁 / 설명
# =======================================
st.markdown("---")
st.markdown("**사용 팁**\n\n"
            "- 임계값을 작게 설정하면(예: 0.001) 더 많은 시퀀스를 '정상'으로 간주합니다.\n"
            "- 임계값을 크게 설정하면(예: 0.05) 더 많은 시퀀스를 '이상'으로 탐지합니다.\n"
            "- `seq_len`을 늘리면(긴 시퀀스) 문맥이 길어져 탐지 성능이 바뀔 수 있습니다.\n"
            "- 전이행렬은 'flag' 기반입니다. 다른 특징(예: service)을 쓰면 결과가 달라집니다.")
