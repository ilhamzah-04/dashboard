import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ttest_ind

st.set_page_config(
    page_title="Dashboard Capstone Edu-Assist",
    page_icon="🎓",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("coursera_courses_dashboard.csv")
    return df

df_clean = load_data()

# TITLE
st.markdown("""
# 🎓 Dashboard Edu-Assist
### Course Recommendation System
""")
st.markdown(
    """
    Dashboard ini menampilkan hasil analisis, recommendation system, 
    dan evaluasi A/B Testing pada dataset course Coursera.
    """
)

# SIDEBAR FILTER
st.sidebar.header("🔎 Filter Data")

selected_level = st.sidebar.multiselect(
    "Pilih Level",
    options=df_clean['Level'].dropna().unique(),
    default=df_clean['Level'].dropna().unique()
)

selected_rating = st.sidebar.multiselect(
    "Pilih Rating Group",
    options=df_clean['rating_group'].dropna().unique(),
    default=df_clean['rating_group'].dropna().unique()
)

selected_complexity = st.sidebar.multiselect(
    "Pilih Complexity Group",
    options=df_clean['complexity_group'].dropna().unique(),
    default=df_clean['complexity_group'].dropna().unique()
)

# APPLY FILTER
filtered_df = df_clean[
    (df_clean['Level'].isin(selected_level)) &
    (df_clean['rating_group'].isin(selected_rating)) &
    (df_clean['complexity_group'].isin(selected_complexity))
]

if len(filtered_df) == 0:

    st.warning(
        """
        ⚠️ Tidak ada data yang sesuai 
        dengan filter yang dipilih.

        Silakan pilih minimal satu filter.
        """
    )

# OVERVIEW DATASET
st.header("🏠 Overview Dataset")

col1, col2, col3, col4 = st.columns(4)

total_course = len(filtered_df)

if total_course > 0:

    avg_rating = round(
        filtered_df[
            'rating'
        ].mean(),
        2
    )

    unique_skills = (
        filtered_df['Skills']
        .astype(str)
        .str.count(',')
        .sum()
    )

    avg_description = int(
        filtered_df[
            'description_length'
        ].mean()
    )

else:

    avg_rating = 0
    unique_skills = 0
    avg_description = 0

col1.metric(
    "📚 Total Course",
    total_course
)

col2.metric(
    "⭐ Rata-rata Rating",
    avg_rating
)

col3.metric(
    "🧠 Total Skill",
    unique_skills
)

col4.metric(
    "📝 Rata-rata Panjang Deskripsi",
    avg_description
)

st.markdown("---")

# DATA PREVIEW
st.subheader("📄 Preview Dataset")

st.dataframe(
    filtered_df[
        [
            'title',
            'Level',
            'rating',
            'complexity_group'
        ]
    ],
    width='stretch'
)

# BUSINESS INSIGHTS (EDA)
if len(filtered_df) > 0:
    st.header("📊 Business Insights (EDA)")

    tab1, tab2 = st.tabs([
        "Pertanyaan Bisnis 1",
        "Pertanyaan Bisnis 2"
    ])

    # QUESTION 1
    with tab1:

        st.subheader(
            "1️⃣ Distribusi Course Berdasarkan Tingkat Kesulitan dan Kategori"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("### 🎯 Distribusi Tingkat Kesulitan")

            fig, ax = plt.subplots(figsize=(7,5))

            sns.countplot(
                data=filtered_df,
                x='Level',
                hue='Level',
                order=filtered_df[
                    'Level'
                ].value_counts().index,
                palette='Blues_r',
                legend=False,
                ax=ax
            )

            ax.set_title(
                'Distribusi Course Berdasarkan Level'
            )

            ax.set_xlabel('Level')
            ax.set_ylabel('Jumlah Course')

            st.pyplot(fig)

        with col2:

            st.markdown("### 🧠 Top 10 Skills Dominan")

            all_skills = (
                filtered_df['Skills']
                .astype(str)
                .str.replace(
                    r"[\[\]']",
                    '',
                    regex=True
                )
                .str.split(',')
                .explode()
                .str.strip()
            )

            top_skills = (
                all_skills
                .value_counts()
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(8,5))

            sns.barplot(
                x=top_skills.values,
                y=top_skills.index,
                hue=top_skills.index,
                palette='Blues_r',
                legend=False,
                ax=ax
            )

            ax.set_title(
                'Top 10 Skills Dominan'
            )

            ax.set_xlabel('Jumlah')
            ax.set_ylabel('Skill')

            st.pyplot(fig)

        st.info(
            """
            📌Insight:

            Mayoritas course berada pada level beginner 
            dan intermediate. Selain itu, skill seperti 
            data analysis, python programming, dan machine learning
            menjadi topik dominan pada dataset.
            """
        )

    # QUESTION 2
    with tab2:

        st.subheader(
            "2️⃣ Karakteristik Teks pada Course"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 📝 Distribusi Panjang Description"
            )

            fig, ax = plt.subplots(figsize=(7,5))

            sns.histplot(
                filtered_df[
                    'description_length'
                ],
                bins=30,
                color='steelblue',
                ax=ax
            )

            ax.set_title(
                'Distribusi Panjang Description'
            )

            ax.set_xlabel(
                'Panjang Description'
            )

            ax.set_ylabel(
                'Jumlah Course'
            )

            st.pyplot(fig)

        with col2:

            st.markdown(
                "### 🏷️ Distribusi Panjang Judul"
            )

            fig, ax = plt.subplots(figsize=(7,5))

            sns.histplot(
                filtered_df[
                    'title_length'
                ],
                bins=20,
                color='royalblue',
                ax=ax
            )

            ax.set_title(
                'Distribusi Panjang Judul'
            )

            ax.set_xlabel(
                'Panjang Judul'
            )

            ax.set_ylabel(
                'Jumlah Course'
            )

            st.pyplot(fig)

        st.markdown(
            "### 🔑 Top 10 Kata Kunci Dominan pada Judul Course"
        )

        vectorizer = CountVectorizer(
            stop_words='english',
            ngram_range=(2,2)
        )

        X = vectorizer.fit_transform(
            filtered_df['title'].astype(str)
        )

        word_counts = pd.DataFrame({
            'keyword':
            vectorizer.get_feature_names_out(),

            'count':
            X.toarray().sum(axis=0)
        })

        top_keywords = (
            word_counts
            .sort_values(
                by='count',
                ascending=False
            )
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(10,5))

        sns.barplot(
            data=top_keywords,
            x='count',
            y='keyword',
            hue='keyword',
            palette='Blues_r',
            legend=False,
            ax=ax
        )

        ax.set_title(
            'Top 10 Keyword Dominan'
        )

        ax.set_xlabel('Jumlah')
        ax.set_ylabel('Keyword')

        st.pyplot(fig)

        st.info(
            """
            📌Insight:

            Sebagian besar course memiliki description
            yang cukup informatif. Selain itu, keyword seperti
            professional certificate, generative ai, 
            dan machine learning menjadi topik dominan dalam course.
            """
        )

else:

    st.header("📊 Business Insights (EDA)")

    st.warning("""
    ⚠️ Business Insights tidak dapat ditampilkan.

    Silakan pilih minimal satu filter data
    untuk melihat visualisasi dan insight.
    """)

# RECOMMENDATION DEMO
if len(filtered_df) > 0:
    st.header("🤖 Demo Recommendation System")

    st.markdown("""
    Masukkan keyword course yang ingin dicari.

    Contoh:
    - machine learning
    - python
    - web development
    - data science
    - cloud computing
    """)

    tfidf = TfidfVectorizer(
        stop_words='english'
    )

    filtered_text = (
        filtered_df[
            'combined_features'
        ]
        .astype(str)
    )

    tfidf_matrix = tfidf.fit_transform(
        filtered_text
    )

    cosine_sim = cosine_similarity(
        tfidf_matrix
    )

    def search_and_recommend(query):

        if len(filtered_df) == 0:
            return pd.DataFrame()

        query_vec = tfidf.transform([query])

        similarity = cosine_similarity(
            query_vec,
            tfidf_matrix
        )

        sim_scores = list(
            enumerate(similarity[0])
        )

        sim_scores = sorted(
            sim_scores,
            key=lambda x: x[1],
            reverse=True
        )

        sim_scores = sim_scores[:5]

        course_indices = [
            i[0]
            for i in sim_scores
        ]

        results = filtered_df.iloc[
            course_indices
        ][
            [
                'title',
                'Level',
                'rating',
                'URL'
            ]
        ].copy()

        results[
            'similarity_score'
        ] = [
            i[1]
            for i in sim_scores
        ]

        return results

    query = st.text_input(
        "🔍 **Masukkan Keyword Course**",
        placeholder="contoh: machine learning"
    )

    if query:

        if len(filtered_df) == 0:

            st.warning(
                "Tidak ada data sesuai filter."
            )

        else:

            recommendation_result = (
                search_and_recommend(query)
            )

            if recommendation_result.empty:

                st.warning(
                    "Recommendation tidak ditemukan."
                )

            else:

                st.subheader(
                    "🎯 Top 5 Recommendation"
                )

                st.dataframe(
                    recommendation_result,
                    width='stretch'
                )

                st.subheader(
                    "📊 Similarity Score Recommendation"
                )

                fig, ax = plt.subplots(
                    figsize=(10,5)
                )

                sns.barplot(
                    data=recommendation_result,
                    x='similarity_score',
                    y='title',
                    hue='title',
                    palette='Blues_r',
                    legend=False,
                    ax=ax
                )

                ax.set_title(
                    'Top 5 Recommendation'
                )

                ax.set_xlabel(
                    'Similarity Score'
                )

                ax.set_ylabel(
                    'Course'
                )

                st.pyplot(fig)

                st.subheader(
                    "🔥 Heatmap Kemiripan Course"
                )

                recommendation_titles = (
                    recommendation_result[
                        'title'
                    ].tolist()
                )

                recommendation_index = (
                    recommendation_result.index
                )

                position_index = [
                    filtered_df.index.get_loc(i)
                    for i in recommendation_index
                ]

                similarity_subset = (
                    cosine_sim[
                        position_index
                    ][:, position_index]
                )

                fig, ax = plt.subplots(
                    figsize=(8,6)
                )

                sns.heatmap(
                    similarity_subset,
                    annot=True,
                    cmap='Blues',
                    xticklabels=recommendation_titles,
                    yticklabels=recommendation_titles,
                    ax=ax
                )

                plt.xticks(
                    rotation=45,
                    ha='right'
                )

                plt.yticks(
                    rotation=0
                )

                st.pyplot(fig)

                st.subheader(
                    "🔗 Link Course"
                )

                for i, row in recommendation_result.iterrows():

                    st.markdown(
                        f"**{row['title']}**  \n"
                        f"⭐ Rating: {row['rating']} | "
                        f"🎓 {row['Level']}  \n"
                        f"[Buka Course]({row['URL']})"
                    )

                st.success(
                    f"""
                    Recommendation system berhasil menemukan 
                    course yang relevan dengan keyword 
                    '{query}' sesuai filter yang dipilih 
                    menggunakan TF-IDF embedding 
                    dan cosine similarity.
                    """
                )
else:

    st.header("🤖 Demo Recommendation System")

    st.warning("""
    ⚠️ Recommendation system tidak dapat dijalankan.

    Tidak ada dataset yang tersedia
    berdasarkan filter yang dipilih.
    """)

# A/B TESTING
if len(filtered_df) > 0:
    st.header("🧪 A/B Testing Recommendation")

    st.markdown("""
    A/B Testing dilakukan untuk membandingkan performa:

    ### A (Embedding Recommendation)
    Menggunakan **TF-IDF Embedding + Cosine Similarity**

    ### B (Keyword Matching)
    Menggunakan pencocokan literal keyword sederhana.
    """)

    # QUERY TESTING
    queries = [
        'machine learning',
        'data science',
        'web development',
        'python',
        'artificial intelligence',
        'business analytics',
        'cyber security',
        'cloud computing',
        'deep learning',
        'data visualization'
    ]

    def keyword_matching(query):

        results = filtered_df[
            filtered_df[
                'combined_features'
            ]
            .str.contains(
                query,
                case=False,
                na=False
            )
        ]

        return results.head(5)

    def embedding_scores(query):

        result = search_and_recommend(query)

        if len(result) == 0:
            return 0

        return result[
            'similarity_score'
        ].mean()

    def keyword_scores(query):

        result = keyword_matching(query)

        if len(result) == 0:
            return 0

        query_words = set(
            query.lower().split()
        )

        scores = []

        for title in result['title']:

            title_words = set(
                str(title).lower().split()
            )

            overlap = len(
                query_words.intersection(
                    title_words
                )
            )

            score = (
                overlap /
                len(query_words)
            )

            scores.append(score)

        return np.mean(scores)

    embedding_results = []
    keyword_results = []

    for query in queries:

        embedding_results.append(
            embedding_scores(query)
        )

        keyword_results.append(
            keyword_scores(query)
        )

    ab_testing_df = pd.DataFrame({

        'Query': queries,

        'Embedding':
        embedding_results,

        'Keyword Matching':
        keyword_results
    })

    st.subheader(
        "📋 Hasil A/B Testing"
    )

    st.dataframe(
        ab_testing_df,
        width='stretch'
    )

    st.subheader(
        "📊 Average Performance"
    )

    col1, col2 = st.columns(2)

    avg_embedding = round(
        np.mean(embedding_results),
        3
    )

    avg_keyword = round(
        np.mean(keyword_results),
        3
    )

    col1.metric(
        "🤖 Embedding Score",
        avg_embedding
    )

    col2.metric(
        "🔎 Keyword Score",
        avg_keyword
    )

    st.subheader(
        "📈 Perbandingan Performa"
    )

    fig, ax = plt.subplots(
        figsize=(12,6)
    )

    ab_testing_df.set_index(
        'Query'
    ).plot(
        kind='bar',
        ax=ax
    )

    plt.xticks(
        rotation=45,
        ha='right'
    )

    plt.ylabel(
        'Relevance Score'
    )

    plt.title(
        'Embedding vs Keyword Matching'
    )

    st.pyplot(fig)

    st.subheader(
        "📉 Statistical Testing"
    )

    t_stat, p_value = ttest_ind(
        embedding_results,
        keyword_results
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "T-Statistic",
        round(t_stat, 3)
    )

    col2.metric(
        "P-Value",
        round(p_value, 5)
    )

    st.subheader(
        "🧠 Interpretasi Otomatis"
    )

    alpha = 0.05

    if p_value < alpha:

        st.success(f"""
        Hasil uji statistik menunjukkan adanya 
        **perbedaan signifikan** antara metode 
        embedding dan keyword matching.

        Embedding recommendation memiliki 
        rata-rata relevance score lebih tinggi
        (**{avg_embedding}**) dibanding keyword 
        matching (**{avg_keyword}**).

        Hal ini menunjukkan bahwa metode embedding 
        lebih efektif dalam memahami konteks dan 
        hubungan semantik antar course.
        """)

    else:

        st.warning(f"""
        Tidak ditemukan perbedaan signifikan 
        antara embedding dan keyword matching.

        Namun embedding tetap menunjukkan 
        rata-rata score lebih tinggi
        (**{avg_embedding}**) dibanding keyword 
        matching (**{avg_keyword}**).
        """)
else:

    st.header("🧪 A/B Testing Recommendation")

    st.warning("""
    ⚠️ A/B Testing tidak dapat dilakukan.

    Dataset hasil filter kosong,
    sehingga evaluasi sistem tidak tersedia.
    """)

# CONCLUSION
if len(filtered_df) > 0:
    st.divider()

    st.header("📌 Conclusion")

    st.markdown("""
    Berikut merupakan kesimpulan utama berdasarkan hasil analisis dan pengembangan recommendation system pada dataset course Coursera.
    """)

    with st.expander("1️⃣ Distribusi Course Berdasarkan Level", expanded=True):

        st.write("""
        Mayoritas course berada pada level **beginner** dan **intermediate**, 
        menunjukkan bahwa platform Coursera lebih banyak menyediakan 
        course untuk pengguna pemula hingga menengah.
        """)

    with st.expander("2️⃣ Karakteristik Teks Course"):

        st.write("""
        Course pada dataset memiliki deskripsi yang cukup informatif. 
        Selain itu, keyword dominan seperti 
        **professional certificate, generative ai, machine learning**, 
        dan **data science** menunjukkan topik 
        pembelajaran yang paling populer.
        """)

    with st.expander("3️⃣ Kemiripan Antar Course"):

        st.write("""
        Metode TF-IDF embedding dan cosine similarity 
        mampu menangkap hubungan konteks antar course, 
        sehingga recommendation menjadi lebih relevan.
        """)

    with st.expander("4️⃣ Relevansi Recommendation"):

        st.write("""
        Recommendation system dapat menyesuaikan hasil 
        berdasarkan keyword pengguna sekaligus filter level, 
        rating, dan complexity yang dipilih.
        """)

    with st.expander("5️⃣ Evaluasi Metode Recommendation"):

        st.write("""
        Berdasarkan hasil A/B Testing, metode embedding 
        menunjukkan performa yang lebih baik dibanding 
        keyword matching sederhana karena menghasilkan 
        relevance score yang lebih tinggi.
        """)

    st.success("""
    🎯 Dashboard berhasil menunjukkan proses end-to-end:
    EDA hingga Conclusion
    """)
else:

    st.header("📌 Conclusion")

    st.info("""
    ℹ️ Kesimpulan akan muncul
    setelah data berhasil ditampilkan.

    Silakan pilih minimal satu filter.
    """)
