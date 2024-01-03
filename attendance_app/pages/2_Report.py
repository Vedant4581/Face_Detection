from Home import st
from Home import pd


st.subheader('Report')

report=pd.read_csv('att_report.csv')
st.write(report)