# from secrets import choice
import pandas as pd
import numpy as np
import squarify
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

import warnings
warnings.filterwarnings('ignore')

from sklearn import metrics
import import_ipynb
import pickle
import streamlit as st
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import Lib
import seaborn as sns 

# 1. Read data
df = pd.read_csv("OnlineRetail.zip", encoding='unicode_escape')
df_show = df.copy()
#--------------
# GUI
st.title("Data Science Projects")
st.header("PROJECT 1 - Customer Segmentation")

# 2. Data pre-processing
# drop rows where Quantity < 0
df = df[df.Quantity >= 0]
# drop rows where CustomerID == null
df = df[df.CustomerID.notnull()]
# drop rows where UnitPrice < 0
df = df[df.UnitPrice >= 0]
# drop duplicated rows
df = df.drop_duplicates()
# Convert column 'InvoiceDate' to datetime datatype
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# convert InvoiceNo data type to integer
df.InvoiceNo = df.InvoiceNo.astype(int)
# Create new column 'Amount'
df['Amount'] = df['Quantity'] * df['UnitPrice']

# Get max date of dataframe
max_date = df['InvoiceDate'].max().date()
# Calculate R, F, M
Recency = lambda x: (max_date - x.max().date()).days
Frequency = lambda x: x.nunique()
Monetary = lambda x: round(sum(x),2)

df_RFM = df.groupby('CustomerID').agg({'InvoiceDate' : Recency,
                                        'InvoiceNo' : Frequency,
                                        'Amount' : Monetary,
                                        })
# Rename column names
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
Monetary_mean = df_RFM.Monetary.mean()

# Create labels for Recency, Frequency, Monetary & Assign them to 4 equal percentile groups and convert their labels from categorical to integer
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=range(4, 0, -1)).astype(int)
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=range(1,5)).astype(int)
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=range(1,5)).astype(int)

# Create new columns R, F, M
df_RFM = df_RFM.assign(R=r_groups, F=f_groups, M=m_groups)

# Remove outliers
data_RFM_no, data_outlier, max_m, max_f, max_r = Lib.drop_outliers(df_RFM)
elite, regular, ghost = Lib.elite_regular_ghost_group(df_RFM, data_outlier, max_m, max_f, max_r, Monetary_mean)
elite["Cluster"] = 4
ghost["Cluster"] = 1
regular["Cluster"] = 3

# Scale data
scaler = MinMaxScaler()
data_RFM_scaled = data_RFM_no.copy()
scaler = scaler.fit(data_RFM_scaled[['Recency','Frequency','Monetary']])
data_RFM_scaled[['Recency','Frequency','Monetary']]= scaler.transform(data_RFM_scaled[['Recency','Frequency','Monetary']])

# 3. Load model
with open('CS_KMean_Scaled_No.pkl', 'rb') as file:  
    kmeans_model = pickle.load(file)

data_RFM_kmeans = data_RFM_no.copy()
data_RFM_kmeans['Cluster'] = kmeans_model.labels_
data_RFM_kmeans2 = data_RFM_kmeans.copy()
data_RFM_kmeans = data_RFM_kmeans.append(elite)
data_RFM_kmeans = data_RFM_kmeans.append(ghost)
data_RFM_kmeans = data_RFM_kmeans.append(regular)

# Calculate average values for each Cluster, and return a size of each segment 
km_agg = Lib.create_df_agg(data_RFM_kmeans, 'Cluster')

# Reset the index
km_agg = km_agg.reset_index(drop=True)
dict_seg = {0:'Gold', 1:'Ghost', 2:'Member', 3:'Deluxe', 4:'Platinum'}
km_agg['Segment_name'] = km_agg['Cluster'].apply(lambda x: dict_seg[x])
data_RFM_kmeans['Segment_name'] = data_RFM_kmeans['Cluster'].apply(lambda x: dict_seg[x])

# #4. Save models
# pkl_filename = "Customer_Segmentation_GMM.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(gmm, file)

# #5. Load models 
# with open(pkl_filename, 'rb') as file:  
#     gmm_model = pickle.load(file)

# GUI
menu = ['Business Objective', 'Build Project', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.image('img1.png')
st.sidebar.image('img3.png')
# st.sidebar.image('img2.png', caption='Customer Segmentation')
if choice == 'Business Objective':
    st.subheader('Business Objective')
    st.write("""
    - Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt. Nhiều khách hàng của công ty là khách hàng bán buôn.
    - Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.
    """)  
    st.code("""=> PROBLEM/ REQUIREMENT: Sử dụng Machine Learning để phân loại các nhóm khách hàng.""")
    st.code("""=> SOLUTION: Phân nhóm khách hàng dựa trên các chỉ số:
            - Recency (R): Thời gian mua hàng gần nhất
            - Frequency (F): Tần suất mua hàng
            - Monetary (M): Tổng chi tiêu""")
    st.image('RFM_solution.jpg')
elif choice == 'Build Project':
    st.subheader('Build Project')
    st.write('### 1. Raw data')
    st.dataframe(df_show.head(5))
    st.dataframe(df_show.tail(5))

    st.write('### 2. RFM data')
    st.dataframe(df_RFM[['Recency','Frequency','Monetary']].head(5))
    st.dataframe(df_RFM[['Recency','Frequency','Monetary']].tail(5))

    st.write('### 3. Visualize R, F, M')
    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    sns.distplot(df_RFM['Recency'])
    plt.subplot(1, 3, 2)
    sns.distplot(df_RFM['Frequency'])
    plt.subplot(1, 3, 3)
    sns.distplot(df_RFM['Monetary'])
    plt.suptitle('Distplot of R, F, M')
    st.pyplot(fig)

    fig = plt.figure(figsize=(10, 3))
    plt.subplot(1, 3, 1)
    plt.boxplot(df_RFM['Recency'])
    plt.subplot(1, 3, 2)
    plt.boxplot(df_RFM['Frequency'])
    plt.subplot(1, 3, 3)
    plt.boxplot(df_RFM['Monetary'])
    plt.suptitle('Boxplot of R, F, M')
    st.pyplot(fig)

    st.write('### 4. Build model & Report')

    # Visualize results
    col1, col2 = st.columns(2)

    with col1:
        fig = plt.figure(figsize=(8, 5))
        count = data_RFM_kmeans.groupby(data_RFM_kmeans['Segment_name']).size()
        fig, ax =  plt.subplots()
        ax.bar(count.index, count.values, color='lightskyblue')
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=90)
        plt.title('Number of Customers of each Segment')
        st.pyplot(fig)

    st.dataframe(km_agg.iloc[:,[0,9,1,2,3,4,5,6,7,8]])
    st.write("""
    Cluster description:
    - Cluster 4 - PLATINUM    : Huge spending or Frequent purchase and High spending
    - Cluster 0 - GOLD        : High spending, frequent purchase and recently buying
    - Cluster 3 - DELUXE      : Medium-high spending, , quite frequent purchase and recently buying
    - Cluster 2 - MEMBER      : Low spending, infrequent purchase and have not purchased for a long time
    - Cluster 1 - GHOST       : Only few times purchase, low spending and have not purchased for a long time
    """)  
    st.write('RESULTS')
    st.dataframe(data_RFM_kmeans.head())
    fig = plt.figure(figsize=(15, 6))
    sns.set_style('darkgrid')
    plt.subplot(1,3,1)
    sns.boxplot(x='Segment_name', y='Recency', data=data_RFM_kmeans)
    plt.title('Recency')
    plt.subplot(1,3,2)
    sns.boxplot(x='Segment_name', y='Frequency', data=data_RFM_kmeans)
    plt.title('Frequency')
    plt.subplot(1,3,3)
    sns.boxplot(x='Segment_name', y='Monetary', data=data_RFM_kmeans)
    plt.title('Monetary')
    plt.tight_layout()
    st.pyplot(fig)

    # Visualize results
    fig = plt.figure(figsize=(10, 5))
    # km_agg2 = km_agg.iloc[:,[0,1,2,3,7,8]]
    st.write('Treemap')
    Lib.treemap_customer_segmentation(km_agg.iloc[:,[0,1,2,3,7,8]],font_size=14)
    st.pyplot(fig)

    # Visualization - 2D Scatter
    st.write('2D scatter plot')
    fig = px.scatter(km_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
           hover_name="Cluster", size_max=80)
    st.plotly_chart(fig)

    # Visualization - 3D scatter
    data_RFM_kmeans2['Monetary'] = np.log(data_RFM_kmeans2['Monetary'])
    data_RFM_kmeans2['Frequency'] = np.log(data_RFM_kmeans2['Frequency'])
    data_RFM_kmeans2['Recency'] = np.log(data_RFM_kmeans2['Recency'])
    # data_RFM_kmeans2.head()
    st.write('3D scatter plot')
    fig = px.scatter_3d(data_RFM_kmeans2, x='Recency', y='Frequency', z='Monetary',
                        color = 'Cluster', opacity=0.5)
    fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
    fig.add_trace(go.Scatter3d(x=np.log(ghost['Recency']), y=np.log(ghost['Frequency']), z=np.log(ghost['Monetary']), 
                          mode='markers',marker=dict(color='green', symbol='cross'), textposition='top left', showlegend=False))
    fig.add_trace(go.Scatter3d(x=np.log(elite['Recency']), y=np.log(elite['Frequency']), z=np.log(elite['Monetary']), 
                          mode='markers',marker=dict(color='red', symbol='cross'), textposition='top left', showlegend=False))
    fig.add_trace(go.Scatter3d(x=np.log(regular['Recency']), y=np.log(regular['Frequency']), z=np.log(regular['Monetary']), 
                          mode='markers',marker=dict(color='orange', symbol='cross'), textposition='top left', showlegend=False))
    st.plotly_chart(fig)

elif choice == 'New Prediction':
    option = st.radio('Select one',['Upload data (Requirement: CustomerID, InvoiceDate, Quantity, UnitPrice)'
                                    ,'Input values of R,F,M'])
    
    if option == 'Upload data (Requirement: CustomerID, InvoiceDate, Quantity, UnitPrice)':
        flag = False
        uploaded_file2 = st.file_uploader('Choose a csv file', type=['csv'])
        if uploaded_file2 is not None:
            data = pd.read_csv(uploaded_file2, encoding='latin-1')
            st.dataframe(data)
            st.write('Data shape:', data.shape)
            flag=True

        if flag:
            st.write('Result (saved to Results.csv):')
            if data.shape[0]>0:
                #Preprocessing 
                data_RFM = Lib.df_RMF_preprocessing(data)
                data_RFM_no, data_outlier = Lib.drop_outliers_predict(data_RFM, max_m, max_f, max_r)
                elite2, regular2, ghost2 = Lib.elite_regular_ghost_group(data_RFM, data_outlier, max_m, max_f, max_r, Monetary_mean)
                data_RFM_scaled = scaler.transform(data_RFM_no[['Recency','Frequency','Monetary']])
                segment = kmeans_model.predict(data_RFM_scaled)
                data_RFM_no['Cluster'] = segment
                if elite2.shape[0]>0:
                    elite2["Cluster"] = 4
                    data_RFM_no = data_RFM_no.append(elite2)
                if ghost2.shape[0]>0:
                    ghost2["Cluster"] = 1
                    data_RFM_no = data_RFM_no.append(ghost2)
                if regular2.shape[0]>0:
                    regular["Cluster"] = 3
                    data_RFM_no = data_RFM_no.append(regular2)
                data_RFM_no['Segment_name'] = data_RFM_no['Cluster'].apply(lambda x: dict_seg[x])
                data_RFM_no = data_RFM_no.sort_values('CustomerID', ascending=True)
                data_RFM_no.to_csv('Results.csv')
                
                st.dataframe(data_RFM_no)
                value_counts = data_RFM_no['Segment_name'].value_counts()
                st.code(value_counts)
                
                #Visualization
                # data_RFM_kmeans2['Monetary'] = np.log(data_RFM_kmeans2['Monetary'])
                # data_RFM_kmeans2['Frequency'] = np.log(data_RFM_kmeans2['Frequency'])
                # data_RFM_kmeans2['Recency'] = np.log(data_RFM_kmeans2['Recency'])
                # fig = px.scatter_3d(data_RFM_kmeans2, x='Recency', y='Frequency', z='Monetary',
                #                     color = 'Cluster', opacity=0.5)
                # fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
                # fig.add_trace(go.Scatter3d(x=np.log(ghost['Recency']), y=np.log(ghost['Frequency']), z=np.log(ghost['Monetary']), 
                #                     mode='markers',marker=dict(color='green', symbol='cross'), textposition='top left', showlegend=False))
                # fig.add_trace(go.Scatter3d(x=np.log(elite['Recency']), y=np.log(elite['Frequency']), z=np.log(elite['Monetary']), 
                #                     mode='markers',marker=dict(color='red', symbol='cross'), textposition='top left', showlegend=False))
                # fig.add_trace(go.Scatter3d(x=np.log(regular['Recency']), y=np.log(regular['Frequency']), z=np.log(regular['Monetary']), 
                #                     mode='markers',marker=dict(color='blue', symbol='cross'), textposition='top left', showlegend=False))
                
                data_RFM_kmeans['Monetary'] = np.log(data_RFM_kmeans['Monetary'])
                data_RFM_kmeans['Frequency'] = np.log(data_RFM_kmeans['Frequency'])
                data_RFM_kmeans['Recency'] = np.log(data_RFM_kmeans['Recency'])                
                fig = px.scatter_3d(data_RFM_kmeans, x='Recency', y='Frequency', z='Monetary',
                                    color = 'Cluster', opacity=0.5)
                fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
                
                fig.add_trace(go.Scatter3d(x=np.log(data_RFM_no['Recency']), y=np.log(data_RFM_no['Frequency']), z=np.log(data_RFM_no['Monetary']), 
                           mode='markers',marker=dict(color='cyan', size=3,line=dict(color='black',width=40), symbol='x'), 
                           textposition='top left', showlegend=False))
                st.plotly_chart(fig)
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            recency = st.number_input('Recency', step=1)
            st.write('The current number is ', recency)
        with col2:
            frequency = st.number_input('Frequency', step=1)
            st.write('The current number is ', frequency)
        with col3:
            monetary = st.number_input('Monetary', min_value=0.00)
            st.write('The current number is ', monetary)
        if st.button('Make Prediction'):
            if (recency>=0) & (frequency>0) & (monetary>0):
                df_rfm = pd.DataFrame([recency,frequency,monetary]).T
                df_rfm.columns = ['Recency','Frequency','Monetary']            
                df_rfm_no, data_outlier = Lib.drop_outliers_predict(df_rfm, max_m, max_f, max_r)
                if df_rfm_no.shape[0]>0:
                    df_rfm_no = scaler.transform(df_rfm_no)
                
                elite, regular, ghost = Lib.elite_regular_ghost_group(df_rfm, data_outlier, max_m, max_f, max_r, Monetary_mean)
                if elite.shape[0]>0:
                        cluster = 4
                        result = 'Platinum'
                elif ghost.shape[0]>0:
                        cluster = 1
                        result = 'Ghost'
                elif regular.shape[0]>0:
                        cluster = 3
                        result = 'Deluxe'
                else:
                    segment2 = kmeans_model.predict(df_rfm_no)
                    cluster = segment2[0]
                    result = dict_seg[cluster]
                st.write('Customer Segment:', cluster, ' - ',result)
                #Visualization 
                data_RFM_kmeans['Monetary'] = np.log(data_RFM_kmeans['Monetary'])
                data_RFM_kmeans['Frequency'] = np.log(data_RFM_kmeans['Frequency'])
                data_RFM_kmeans['Recency'] = np.log(data_RFM_kmeans['Recency'])                
                fig = px.scatter_3d(data_RFM_kmeans, x='Recency', y='Frequency', z='Monetary',
                                        color = 'Cluster', opacity=0.5)
                fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
                
                # fig = px.scatter_3d(data_RFM_kmeans2, x='Recency', y='Frequency', z='Monetary',
                #             color = 'Cluster', opacity=0.5)
                # fig.update_traces(marker=dict(size=5),selector=dict(mode='markers'))
                fig.add_trace(go.Scatter3d(x=np.log([recency]), y=np.log([frequency]), z=np.log([monetary]), 
                                mode='markers',marker=dict(color='cyan', line=dict(color='black',width=40), symbol='x',size=8), 
                                textposition='top left', showlegend=False))
                st.plotly_chart(fig)
            else: 
                st.code(
                """ValueError: \n- Valid values: Recency>=0, Frequency>0, Monetary>0""")
                
