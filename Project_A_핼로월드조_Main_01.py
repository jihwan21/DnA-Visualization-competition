#!/usr/bin/env python
# coding: utf-8

# # Table of contents
# 
# * [Preprocessing](#Preprocessing)
#     + [Missing Value Handling](#Missing-Value-Handling)
#     + [Remove duplicates](#Remove-duplicates)
#     + [Outlier Handling](#Outlier-Handling)
# * [Analyze](#Analyze)
#     + [1. 서울시 자치구별 학원, 교습소 수](#1.-서울시-자치구별-학원,-교습소-수)
#     + [2. 서울시 자치구별 학원, 교습소 정원 수(온라인, 원격 학원/교습소 포함)](#2.-서울시-자치구별-학원,-교습소-정원-수(온라인,-원격-학원/교습소-포함))
#     + [3. 서울시 학군별 학원/교습소 & 학교 수](#3.-서울시-학군별-학원/교습소-&-학교-수)
#     + [4. 서울시 학군별 학원/교습소 학생 수 & 학교 학생 수](#4.서울시-학군별-학교-학생-수-&-학원/교습소-학생-수)
# * [Visualize](#Visualize)
#     + [1. 서울시 자치구별 학원, 교습소 수 시각화](#1.-서울시-자치구별-학원,-교습소-수-시각화)
#     + [2. 서울시 학군별 학원/교습소 & 학교 수 시각화](#2.-서울시-학군별-학원/교습소-&-학교-수-시각화)
#     + [3. 학군별 학원/교습소 학생 수(온라인, 원격 제외) & 학교 학생 수 시각화](#3.-학군별-학원/교습소-학생-수(온라인,-원격-제외)-&-학교-학생-수-시각화)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rc('font', family='malgun gothic')
plt.rc('axes', unicode_minus=False)


# ### Import Data

# In[2]:


# Main Data
academy = pd.read_csv('./data/서울특별시 학원 교습소정보.csv')
school = pd.read_csv('./data/서울특별시 학교 기본정보.csv')
school_cnt = pd.read_excel('./data/2021 서울시 학급당 학생수 통계.xlsx') # 외부데이터
# 전처리 한 데이터를 시트별로 통합 / 그 중 '학원교습소정보'와 '학교기본정보' 시트 사용
df = pd.read_excel('./data/helloworld_data_set.xlsx', sheet_name=['학원교습소정보', '학교기본정보'] )
locals().update(df)


# ### Preprocessing
# 전처리 수행(결측값 처리, 중복값 처리, 이상치 처리 등)

# In[3]:


academy.head()


# In[4]:


# Dtype 확인
academy.info()


# In[5]:


# 열별 결측치 개수 확인
academy.isnull().sum()


# ### Missing Value Handling

# In[6]:


# 도로명상세주소 + 도로명주소 + 교습계열명 + 교습과정명 + 일시수용능력인원합계 결측치 포함 행 제거
academy = academy.dropna(subset=['도로명상세주소', '교습계열명', '일시수용능력인원합계']) 


# In[7]:


academy['행정구역명'] = academy['행정구역명'].fillna(academy['도로명주소'].str.split(' ').str[1])
academy['인당수강료내용'] = academy['인당수강료내용'].fillna('알수없음')
academy['기숙사학원여부'] = academy['기숙사학원여부'].fillna('알수없음')
academy['휴원시작일자'] = academy['휴원시작일자'].fillna(0)
academy['휴원종료일자'] = academy['휴원종료일자'].fillna(0)


# In[8]:


# '서울특별시마포구'같이 띄어쓰기가 안 되어 있는 데이터가 있어 '서울특별시'를 제거하고 앞의 공백을 제거
academy['도로명주소'] = academy['도로명주소'].str.replace('서울특별시', '')
academy['도로명주소'].str.lstrip().tolist()


# In[9]:


# 결측치 처리 확인
academy.isnull().sum()


# ### Remove duplicates

# In[10]:


academy.duplicated().sum() # 중복값 1개 확인


# In[11]:


academy.shape # 중복값 제거 전 행,열 개수


# In[12]:


academy.drop_duplicates(inplace=True) # 중복값 제거


# In[13]:


academy.shape # 중복값 제거 후 행, 열 개수


# ### Outlier Handling

# In[14]:


# 이상치 처리 전
academy.describe()


# In[15]:


academy['휴원종료일자'].unique() # 이상치 99999999


# In[16]:


academy.query('휴원종료일자 == 99999999') # 이상치 포함 행 확인 / 휴원시작일자 0이므로 휴원종료일자도 0으로 대체


# In[17]:


# 휴원종료일자 이상치 99999999를 0으로 대체 (해당 행 휴원시작일자 = 0)
academy['휴원종료일자'] = academy['휴원종료일자'].replace(99999999, 0) 

# 등록일자 오타 수정
academy['등록일자'] = academy['등록일자'].replace(21060906, 20160906) 
academy['등록일자'] = academy['등록일자'].replace(2210707, 20210707)


# In[18]:


academy.select_dtypes('object').describe() # 이상치 없음


# In[19]:


# 이?미용 -> 이·미용, 보습?논술 -> 보습·논술 / 데이터 출처 찾아서 ?를 ·로 수정
academy['교습과정명'] = academy['교습과정명'].str.replace('?', '·') 
academy['교습계열명'] = academy['교습계열명'].str.replace('?', '·') 


# In[20]:


# 일시수용능력인원합계 9999명 이상을 이상치로 보고 0으로 처리 / 9999명 이상 학원/교습소들은 온라인, 원격으로 수업or병행 진행
c1 = academy['일시수용능력인원합계'] < 9999
academy['일시수용능력인원합계'] = academy['일시수용능력인원합계'].where(c1, 0)

# 정원합계 50000명 이상을 이상치로 보고 해당 행 삭제(50000명 이상인 학원/교습소 = 원격, 온라인 학원)
c2 = academy['정원합계'] < 50000
academy['정원합계'] = academy['정원합계'].where(c2, np.nan)
academy = academy.dropna()

# 이상치 처리 후
academy.describe()


# In[21]:


# 등록일자, 개설일자 Dtype object에서 datetime으로 변경
academy['등록일자'] = pd.to_datetime(academy['등록일자'], format='%Y%m%d')
academy['개설일자'] = pd.to_datetime(academy['개설일자'], format='%Y%m%d')


# In[22]:


# Dtype 변경 후
academy.info()


# In[23]:


# 전처리와 이상치 처리 후 인덱스 번호 재설정
academy.reset_index(drop=True, inplace=True)


# ### Analyze
# 데이터 분석(상관 관계, 추세, 변동 등)

# #### 1. 서울시 자치구별 학원, 교습소 수

# In[24]:


# '학원/교습소' column에서 학원과 교습소로 데이터 구분
edu_institute_01 = academy[academy['학원/교습소'] == '학원'] # = academy.groupby('학원/교습소').get_group('학원')
edu_institute_02 = academy[academy['학원/교습소'] != '학원'] # = cademy.groupby('학원/교습소').get_group('교습소')

# column 이름 재설정
edu_institute_01 = edu_institute_01.rename(columns={'학원/교습소':'학원'})
edu_institute_01 = edu_institute_01.rename(columns={'정원합계':'학원 정원합계'})
edu_institute_02 = edu_institute_02.rename(columns={'학원/교습소':'교습소'})
edu_institute_02 = edu_institute_02.rename(columns={'정원합계':'교습소 정원합계'})


# In[25]:


academy_df1 = edu_institute_01.groupby('행정구역명')[['학원']].count() # 자치구별 학원 수
academy_df2 = edu_institute_02.groupby('행정구역명')[['교습소']].count() # 자치구별 교습소 수
df1 = pd.concat([academy_df1,academy_df2],axis=1).sort_values(by='학원', ascending=False) ; df1
# 상위 4개 학원 교습소 밀집 자치구 = 강남구, 서초구, 송파구, 양천구


# #### 2. 서울시 자치구별 학원, 교습소 정원 수(온라인, 원격 학원/교습소 포함)

# - 학원 학생 수 1위인 강남구는 학원 수도 1위이다. 그러나 두 번째로 학원 학생 수가 많은 동작구는 학원 수 순위 10위이다.

# In[26]:


# 자치구별 학원과 교습소 학생 수 합계
academy_df3 = edu_institute_01.groupby('행정구역명')[['학원 정원합계']].sum()
academy_df4 = edu_institute_02.groupby('행정구역명')[['교습소 정원합계']].sum()
# 학원 정원합계를 기준으로 내림차순 정렬
df2 = pd.concat([academy_df3,academy_df4],axis=1).sort_values(by='학원 정원합계', ascending=False) ; df2


# #### 3. 서울시 학군별 학원/교습소 & 학교 수

# In[27]:


# column = 자치구, 학원/교습소, 학원학생수 인덱싱하여 edu_df 생성
col01 = {'자치구': 학원교습소정보['자치구'],
       '학원/교습소': 학원교습소정보['학원/교습소'],
        '학원학생수': 학원교습소정보['정원합계']}
edu_df01 = pd.DataFrame(col01) ; edu_df01


# In[28]:


# column = 자치구, 학교종류명 인덱싱하여 school_df 생성
col02 = {'자치구': 학교기본정보['행정구역명'],
       '학교종류명': 학교기본정보['학교종류명']}
school_df = pd.DataFrame(col02) ; school_df


# In[29]:


# edu_df 자치구별 해당하는 학군열 추가
school_district_lst = ['1학군', '2학군', '3학군', '4학군','5학군', '6학군', '7학군', '8학군', '9학군', '10학군', '11학군']
gu_lst = [['동대문구', '중랑구'], ['마포구', '서대문구', '은평구'], ['구로구', '금천구', '영등포구'], ['노원구', '도봉구'], 
                ['용산구', '종로구', '중구'], ['강동구', '송파구'], ['강서구', '양천구'], ['강남구', '서초구'], ['관악구', '동작구'], 
                ['광진구', '성동구'], '강북구', '성북구']
col = edu_df01['자치구']

for i in range(11):
    globals()['edu_district_'+str(i+1)] = edu_df01.query("@col in @gu_lst[@i]")
    

edu_data_lst = [edu_district_1, edu_district_2, edu_district_3, edu_district_4, edu_district_5, edu_district_6, 
            edu_district_7, edu_district_8, edu_district_9, edu_district_10, edu_district_11]


for k, v in zip(edu_data_lst, school_district_lst):
    k.insert(0, '학군명', np.full([len(k['자치구'])], v))
    
    
edu_district01 = pd.concat([edu_district_1, edu_district_2, edu_district_3,  edu_district_4,  edu_district_5, 
                          edu_district_6, edu_district_7, edu_district_8, edu_district_9, edu_district_10, edu_district_11])
edu_district01


# In[30]:


# school_df 자치구별 해당하는 학군열 추가
col = school_df['자치구']

for i in range(11):
    globals()['school_district_'+str(i+1)] = school_df.query("@col in @gu_lst[@i]")


school_data_lst = [school_district_1, school_district_2, school_district_3, school_district_4, school_district_5, school_district_6, 
            school_district_7, school_district_8, school_district_9, school_district_10, school_district_11]


for k, v in zip(school_data_lst, school_district_lst):
    k.insert(0, '학군명', np.full([len(k['자치구'])], v))
    
    
school_district = pd.concat([school_district_1, school_district_2, school_district_3, school_district_4, school_district_5, 
                             school_district_6, school_district_7, school_district_8, school_district_9, school_district_10, 
                             school_district_11])
school_district


# In[31]:


# 학군별 학원/교습소 수 & 학원학생수
my_dict = {'학원/교습소':'count',
           '학원학생수': 'sum'}
edu_district01 = edu_district01.groupby(['학군명']).agg(my_dict) ; edu_district01


# In[32]:


# 학군별 학교 수
school_district = school_district.groupby('학군명')[['학교종류명']].count() ; school_district


# In[33]:


# column 이름 변경(학교종류명 > 학교수)
school_district.rename(columns={'학교종류명':'학교수'}, inplace=True) 


# In[34]:


# 학군별 학원/교습소 수와 학교수 merge
edu_school = edu_district01[['학원/교습소']].merge(school_district, on='학군명') ; edu_school


# #### 4. 서울시 학군별 학원/교습소 학생 수 & 학교 학생 수

# - 학군별 학교 학생 수

# In[35]:


# 유치원~고등학교 학생 수 합한 열 생성
school_cnt['합계'] = school_cnt['유치원']+school_cnt['초등학교']+school_cnt['중학교']+school_cnt['고등학교']

# school_cnt 데이터프레임에서 '지역', '합계'열 인덱싱하고 ' 합계'열을 기준으로 내림차순 정렬하여 sch_cnt 데이터프레임 생성
sch_cnt = school_cnt[['지역', '합계']].sort_values(by='합계', ascending=False).reset_index(drop=True)

# 열 이름 변경
sch_cnt.rename(columns = {'지역':'자치구', '합계':'학교학생수'}, inplace=True)

# sch_cnt 자치구별 해당하는 학군열 추가
col = sch_cnt['자치구']

for i in range(11):
    globals()['school_district_'+str(i+1)] = sch_cnt.query("@col in @gu_lst[@i]")


school_data_lst = [school_district_1, school_district_2, school_district_3, school_district_4, school_district_5, school_district_6, 
            school_district_7, school_district_8, school_district_9, school_district_10, school_district_11]


for k, v in zip(school_data_lst, school_district_lst):
    k.insert(0, '학군명', np.full([len(k['자치구'])], v))
    
    
sch_district = pd.concat([school_district_1, school_district_2, school_district_3, school_district_4, school_district_5, 
                             school_district_6, school_district_7, school_district_8, school_district_9, school_district_10, 
                             school_district_11])
sch_district


# In[36]:


# 학군별 학교학생수
sch_district = sch_district.groupby('학군명')[['학교학생수']].sum() ; sch_district


# - 학군별 학원 학생 수 (온라인, 원격 학원 제외)

# In[37]:


# 원격, 온라인 학원 제외
학원교습소정보 = 학원교습소정보[~학원교습소정보['학원명'].str.contains('온라인')] # 온라인 in 학원명 행 제거 
학원교습소정보 = 학원교습소정보[~학원교습소정보['학원명'].str.contains('원격')] # 원격 in 학원명 행 제거
학원교습소정보 = 학원교습소정보.rename(columns = {'정원합계':'학원/교습소학생수'}) # 열 이름 변경


# In[38]:


# column = '자치구', '학원학생수' 인덱싱하여 edu_df2 데이터프레임 생성
edu_df02 = 학원교습소정보[['자치구', '학원/교습소학생수']]

# edu_df2 자치구별 해당하는 학군열 추가
col = edu_df02['자치구']

for i in range(11):
    globals()['edu_district_'+str(i+1)] = edu_df02.query("@col in @gu_lst[@i]")
    

edu_data_lst = [edu_district_1, edu_district_2, edu_district_3, edu_district_4, edu_district_5, edu_district_6, 
            edu_district_7, edu_district_8, edu_district_9, edu_district_10, edu_district_11]


for k, v in zip(edu_data_lst, school_district_lst):
    k.insert(0, '학군명', np.full([len(k['자치구'])], v))
    
    
edu_district02 = pd.concat([edu_district_1, edu_district_2, edu_district_3,  edu_district_4,  edu_district_5, 
                          edu_district_6, edu_district_7, edu_district_8, edu_district_9, edu_district_10, edu_district_11])
edu_district02


# In[39]:


# 학군별 학원학생수(온라인, 원격 제외)
edu_district02 = edu_district02.groupby('학군명')[['학원/교습소학생수']].sum() ; edu_district02


# In[40]:


# sch_deu = 학군별 학교학생수 & 학원학생수(온라인,원격 제외)
# merge / '학원학생수' 열을 기준으로 내림차순 정렬
sch_edu = sch_district[['학교학생수']].merge(edu_district02, on='학군명')
sch_edu.sort_values(by='학원/교습소학생수', ascending=False, inplace=True) ; sch_edu 

# 8학군 소재 학교에 다니는 학생보다 학원/교습소에 다니는 학생 수가 많음을 확인


# ### Visualize
# 분석 과정 또는 결과 시각화

# #### 1. 서울시 자치구별 학원/교습소 수 시각화

# In[52]:


# 서울시 자치구별 학원/교습소 수 시각화
plt.figure(figsize=(12, 15))
academy.groupby('행정구역명')['학원/교습소'].count().sort_values().plot(kind='barh', color='royalblue')

plt.yticks(fontsize=12)
plt.title('서울시 자치구별 학원/교습소 수')
plt.xlabel('학원/교습소 수')
plt.show()


# - 서울시 자치구별 학원/교습소 수 지도 시각화 01

# In[42]:


import requests
import json

# 서울 행정구역 json raw파일(githubcontent)
r = requests.get('https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json')
c = r.content
seoul_geo = json.loads(c)

map = folium.Map(
    location=[37.559819, 126.963895],
    zoom_start=11,
    tiles='cartodbpositron'
)

folium.GeoJson(
    seoul_geo,
    name='지역구'
).add_to(map)

academy_group_data = academy.groupby('행정구역명')['학원/교습소'].count().sort_values()

map.choropleth(geo_data=seoul_geo,
             data=academy_group_data, 
             fill_color='YlOrRd',
             fill_opacity=0.5,
             line_opacity=0.2,
             key_on='properties.name',
             legend_name="서울시 자치구별 학원/교습소 수"
            )

map


# - 서울시 자치구별 학원/교습소 수 지도 시각화 02

# In[43]:


academy_df = pd.DataFrame(academy_group_data).reset_index()
academy_df = academy_df.rename(columns={'행정구역명':'name'})


# In[44]:


#geometry_gj = json.load(open('HangJeongDong_ver20220309.geojson', encoding='utf-8'))
r = requests.get('https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json')
c = r.content
seoul_geo = json.loads(c)

fig = px.choropleth(academy_df, geojson=seoul_geo, locations='name', color='학원/교습소',
                                color_continuous_scale='Blues',
                                featureidkey='properties.name')
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text='서울시 자치구별 학원/교습소 수', title_font_size=20)


# #### 2. 서울시 학군별 학원/교습소 & 학교 수 시각화

# In[45]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=edu_district01.index, y=edu_district01['학원/교습소'], name="학원/교습소 수"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=school_district.index, y=school_district['학교수'], name="학교 수"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="서울시 학군별 학원/교습소 & 학교 수"
)

# Set x-axis title
fig.update_xaxes(title_text="학군명")

# Set y-axes titles
fig.update_yaxes(title_text="학원/교습소 수", secondary_y=False)
fig.update_yaxes(title_text="학교 수", secondary_y=True)

fig.show()


# In[46]:


# 학군별 학원/교습소 수와 학교 수 상관관계 그래프
sns.jointplot(x='학원/교습소', y='학교수', kind='reg', height=10, data=edu_school)
# 시각화 결과 양의 상관관계를 가짐을 알 수 있음
# 학원/교습소 수 약 2,500 부터는 신뢰구간이 넓어짐


# In[47]:


# heatmap
plt.subplots(figsize=(5,5))
sns.heatmap(edu_school.corr(), annot=True, fmt='0.2f', cmap="RdYlGn", linewidths=0.2)
# 학원/교습소 수와 학교 수 상관계수 0.65


# #### 3. 학군별 학원/교습소 학생 수(온라인, 원격 제외) & 학교 학생 수 시각화

# In[48]:


# 학군별 학원/교습소 학생 수 / 학교 학생 수 비율 시각화
import plotly.express as px

sch_edu['비율'] = sch_edu['학원/교습소학생수'] / sch_edu['학교학생수']

fig = px.line(sch_edu, x=sch_edu.index, y='비율', title='학군별 학원/교습소 학생 수 / 학교 학생 수 비율')
fig.show()
# 8, 9 학군이 학교 학생 수 대비 학원/교습소 학생 수가 많음
# 비율이 제일 큰 9학군은 학교 학생 수보다 학원 학생 수가 타 학군 대비 상대적으로 많음을 유추할 수 있음


# In[49]:


# 이중 Y축 선 그래프로 시각화
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=sch_edu.index, y=sch_edu['학교학생수'], name="학교 학생 수"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=sch_edu.index, y=sch_edu['학원/교습소학생수'], name="학원/교습소 학생 수"),
    secondary_y=True,
)

fig.update_layout(
    title_text="서울시 학군별 학원/교습소 & 학교 학생 수"
)

fig.update_xaxes(title_text="학군명")

fig.update_yaxes(title_text="학원/교습소 학생 수", secondary_y=False)
fig.update_yaxes(title_text="학교 학생 수", secondary_y=True)

fig.show()

# 8, 9학군이 타 학군에 비해 학원/교습소 학생 수가 압도적으로 많음 > 타 학군 소재 학교 학생들이 8, 9 학군 소재 학원에 등원함을 유추
# 위 비율 시각화에서 유추한대로 9학군은 학교 학생 수보다 학원/교습소 학생 수가 타 학군 대비 상대적으로 많음을 볼 수 있음


# In[50]:


# 학군별 학원/교습소 힉생 수와 학교 학생 수 상관관계 그래프
sns.jointplot(x='학교학생수', y='학원/교습소학생수', kind='reg', height=10, data=sch_edu)
# 양의 상관관계를 보이나 전반적으로 신뢰구간이 넓음


# In[51]:


# 학교 학생 수와 학원/교습소 학생 수의 상관계수 0.33
plt.subplots(figsize=(5,5))
sns.heatmap(sch_edu[['학교학생수', '학원/교습소학생수']].corr(), annot=True, fmt='0.2f', cmap="RdYlGn", linewidths=0.2)

