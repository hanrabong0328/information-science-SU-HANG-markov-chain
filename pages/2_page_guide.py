import streamlit as st
import pandas as pd

st.set_page_config(page_title="IDS 설명 페이지", layout="wide")
st.title("📘 마르코프 체인 IDS 설명")

st.header("1️⃣ 사이트 목적")
st.write("""
이 앱은 정상 네트워크 트래픽 데이터를 기반으로 **연속 이벤트 전이 패턴**을 학습하고,
새로운 이벤트에서 이상 징후를 탐지합니다.
""")

st.header("2️⃣ 주요 용어")
st.write("""
- **연속 이벤트**: 네트워크에서 연속적으로 발생한 연결 상태 묶음  
- **연결 상태(flag)**: 하나의 연결이 정상인지 이상인지 표시  
- **상태 전이 확률**: 현재 상태 다음에 어떤 상태가 나올 확률
""")

st.header("3️⃣ 전이행렬 설명")
st.write("""
- **행(row)**: 현재 상태  
- **열(column)**: 다음 상태  
- **값(value)**: 정상 연결에서 발생할 확률  
- 예: S1 → S2 = 0.3 → 정상 연결에서 S1 다음에 S2가 30% 확률로 발생
""")

example_matrix = pd.DataFrame(
    [[0.4,0.3,0.3],
     [0.2,0.5,0.3],
     [0.1,0.3,0.6]],
    columns=["S1","S2","S3"],
    index=["S1","S2","S3"]
)
st.dataframe(example_matrix)

st.header("4️⃣ 그래프 설명")
st.write("""
- 빨간 점: 이상 이벤트 탐지  
- 파란 점: 정상 이벤트  
- x축: 이벤트 묶음 번호  
- y축: 평균 전이 확률
""")
