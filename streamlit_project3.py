import streamlit as st

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from gensim import corpora, models, similarities
import jieba
import re
from six import iteritems


#load data
@st.cache(suppress_st_warning=True)
def load_data(data):
    df = pd.read_csv(data)
    return df
@st.cache(suppress_st_warning=True)
def load_stopword(STOP_WORDS):
    with open(STOP_WORDS, 'r', encoding = 'utf-8') as file:
        stop_words = file.read()
    stop_words = stop_words.split('\n')
    return stop_words

#vectorize & cosime
@st.cache(suppress_st_warning=True, allow_output_mutation= True)
def tfidf_text_to_cosine(df, stop_words, item_id, num):
    tf = TfidfVectorizer(analyzer = 'word', min_df = 0, stop_words= stop_words)
    tfidf_matrix= tf.fit_transform(df['product_infor_tokenize'])
    cosine_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    results = {}
    for idx, row in df.iterrows():    
        similar_indices = cosine_matrix[idx].argsort()[:-10:-1]
        similar_items = [(cosine_matrix[idx][i], df['item_id'][i]) for i in similar_indices]
        results[row['item_id']] = similar_items[1:]
    
    info = df.loc[df['item_id'] == item_id]['name'].to_list()[0].split('-')[0]
    print('Recommending ' + str(num) + 'products similar to ' + item(item_id, df) + '...')
    print('-'*40)
    recs = results[item_id][:num]
    product_lst = []
    for rec in recs:
        #st.write("Product id: ", rec[1])
        #product_info(rec[1] ,df)
        #st.write('Recommend: '+ item(rec[1], df) + '(score: ' + str(rec[0]) + ')')
        product_lst.append(rec[1])
    return product_lst

@st.cache(suppress_st_warning=True)
def item(id, df):
    return df.loc[df['item_id'] == id]['name'].to_list()[0].split('-')[0]

#gensim
@st.cache(suppress_st_warning=True, allow_output_mutation= True)
def gensim_recommendation(df, stop_words, product_ID, n):
    product_infor_split = [[text for text in x.split()] for x in df['product_infor_tokenize']]
    product_infor_split_re = [[t.lower() for t in text if not t in [' ', ',', '.', '...', '-', ':', ';', '?', '\d+']] for text in product_infor_split]
    #obtain the number of features based on dictionary: use corpora.Dictionary
    dictionary = corpora.Dictionary(product_infor_split_re)
    #get stopwords from dictionary
    stop_ids = [dictionary.token2id[stopword] for stopword in stop_words if stopword in dictionary.token2id]
    #get words occur only 1 (frequency = 1)
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    #remove stopwords
    dictionary.filter_tokens(stop_ids + once_ids)
    # remove gaps in id sequence after words that were removed
    dictionary.compactify()
    #number of features (word) in dictionary
    feature_cnt = len(dictionary.token2id)
    #obtain corpus based on dictionary (dense matrix)
    corpus = [dictionary.doc2bow(text) for text in product_infor_split_re]
    tfidf = models.TfidfModel(corpus)
    index= similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = feature_cnt)
    
    #Recommend for specific user
    product_result= df[df.item_id == product_ID].head(1)    
    product_result= df[df.item_id == product_ID].head(1)
    view_product = product_result['product_infor_tokenize'].to_string(index = False)
    view_product = view_product.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    sim = index[tfidf[kw_vector]]
    
    #print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
    
    df_result = pd.DataFrame({'id': list_id,
                             'score': list_score})
    
    #five highest scores
    five_highest_score = df_result.sort_values(by = 'score', ascending = False).head(n +1)
    idToList = list(five_highest_score['id'])
    
    product_find = df[df.index.isin(idToList)]
    result = product_find[['item_id', 'name', "price", "product_type", "url"]]
    result = pd.concat([result, five_highest_score], axis = 1).sort_values(by = 'score', ascending = False)
    result = result[result.item_id != product_ID]
    return list(result["item_id"].values)

#create product infomation

def product_info(productId, data):
    product = data[data["item_id"] == productId]
    st.write(product["group"].values[0])
    col1, col2 = st.columns(2)
    with col1:
        st.image(product["image"].values[0])
    with col2:
        st.write("##### [" + product["name"].values[0] + "] (https://tiki.vn/" + product["url"].values[0][8:] + ")")
        st.write("Product type: ", product["product_type"].values[0])
        st.write("Brand:", product["brand"].values[0], "|", "Rating:", str(product["rating"].values[0]))
        st.write("Actual price: ", str(product["list_price"].values[0]), "₫")
        st.write("Sell price: ", str(product["price"].values[0]), "₫ (-", str(product["discount"].values[0]*100), "%)")



#Load json data
@st.cache(suppress_st_warning=True)
def load_json(data):
   df = pd.read_json(data)
   return df

#get als_recommendation
@st.cache(suppress_st_warning=True)
def als_recommendation(customerID, data_als, data):
    als_result = data_als[data_als["customer_id"] == customerID]
    result = pd.merge(als_result, data, left_on = "product_id", right_on = "item_id", how = "left")
    #return result[["customer_id", "product_id", "name", "price", "brand", "product_type", "product_name", "url", "image"]]
    return list(result["product_id"])

#GUI


def main():
    st.set_page_config(page_title= "Recommendation App", layout = "wide")
    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        st.write("")

    with col2:
        st.sidebar.image("logo.png", width=130)

    with col3:
        st.write("")
    st.sidebar.write("### Project 3 - Recommendation System")
    menu = ["Home", "Content Based Recommended - Cosine", "Content Based Recommended - Gensim", 
            "Colaborative Filtering - ALS" , "About"]
    choice = st.sidebar.selectbox("",menu)

    #load product dataframe
    df = load_data("product_preprocess.csv")
    #load stopwords
    stop_words = load_stopword("vietnamese-stopwords.txt")
    #load spark json
    als_df = load_json("user_recs.json")

    if choice == "Home":
        st.title("Home")
        st.subheader("Business Understanding")
        st.write("""- Two datasets given: product (product information) & review (product review from users).
- Build the recommendation model for dataset listed above.
- There is 2 way for building recommendation system: Content based/Collaborarive filtering.""")
        st.image("ecommerce.jpg")
        st.subheader("Data Understanding")
        st.write("""- Data was minned from TIKI site contain 4000 record for product dataset and 360k record for the review dataset.
- Those product from variety type of technoligy field so there are many different way for approaching the recommendation model.
- Main features in the two dataset is belonged to NPL part, so there is a bunch of task for preprocessing those features into categorical feature, some numerical exist in those dataset contain: product/reviewer ID, rating.""")
        st.image("content_based.jpg")

    elif choice == "Content Based Recommended - Cosine":
        st.title("Content-Based with Cosine Similarity")
        st.subheader("Recommended Products")
        col1, col2= st.columns(2)
        with col1:
            search_term = st.selectbox("Select Product ID", df["item_id"].unique())
        with col2:
            num_of_rec = st.number_input("How many product recommended?", 4, 20, 7)
        st.write("#### Product information")
        product_info(int(search_term), df)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    for i in tfidf_text_to_cosine(df, stop_words, search_term, num_of_rec):
                        st.write(product_info(i, df))
                except:
                    st.write("Not Found!")
    elif choice == "Content Based Recommended - Gensim":
        st.title("Content Based with Gensim")
        st.subheader("Recommended Products")
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.selectbox("Select Product ID", df["item_id"].unique())
        with col2:
            num_of_rec = st.number_input("How many product recommended? ", 4, 20, 7)
        st.write("#### Product Information")
        product_info(search_term, df)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    for i in gensim_recommendation(df, stop_words, search_term, num_of_rec):
                        product_info(i, df)
                except:
                    st.write("Not Found!")
    
    elif choice == "Colaborative Filtering - ALS":
        st.title("Collaborated Based with ALS")
        st.subheader("Recommended Product")
        search_term = st.selectbox("Select CustomerID", als_df["customer_id"].unique())
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    st.write("### Products you may like")
                    for i in als_recommendation(search_term ,als_df, df):
                        product_info(i, df)
                except:
                    st.write("Not found!")
            else:
                pass
    
    else:
        st.title("About")
        st.write("### Data Understading")
        st.dataframe(df[["item_id", "name", "rating", "brand", "product_type","price", "list_price", "discount"]].head(10))
        st.write("#### Product type is almost technologies related products")
        st.image("type_bar.png")
        st.write("""- The data mostly contain technology product from different segmentation (from cheap to expensive) and the low price accounted the most.
- There is a balance data between technology product, but it is imbalance with some very small amount of product such as car/bike, house/life, etc because those type are very small, but it acceptable.""")
        st.write("#### The more promotion, the more cheaper the price is")
        st.image("price_factor.png")
        st.write("""- As I said above, the data contain mostly popular and mid-range part from variety brands. Some of high-class brand with their high price product can seem as outliers but it does not affect into the recommendation system models.
- The OEM ,Logitech, TP-Link, etc are some a top middle-class brand that have a large amount of product appear in the dataset.""")
        st.write("#### The cheaper the price is, the more discount campaign the Brand will release")
        st.image("discount_factor.png")
        st.write("""- The OEM and Sandisk brand have the most promotion campaign release with discount mean above 45%.
- Most brand do have a discount offer from 30% - 35% and they are middle-class and mostly high-class brand.
- High-class brand Apple have really small disount offer.""")
        st.write("")
        st.write("### Model Information")
        st.write("""Recommender System used to be an important application in ecommerce field.
        The model recommend relevant item for users (film, product, document, etc)""")
        col1, col2 = st.columns(2)
        with col1:
            st.write("##### [Colaborative-Based Filtering] (https://developers.google.com/machine-learning/recommendation/collaborative/basics)")
            st.write("""- Colaborative-Based Filtering create recommendation based on customer behavior in interacting with products. 
            It used 'wisdom of the crowd' for recommending items for users.""")
            st.write("""- It is more popular than Content-Based model because of better result and easy to learn by itself.
            The model will easily recommend something new for users just based on the behavior of some similar users.
- And because Collaborative used customer behavior data so the result is really big so it related to Big data field and need a lot of technique to optimize time execution.""")
        with col2:
            st.write("##### [Content-Based Filtering](https://developers.google.com/machine-learning/recommendation/content-based/basics)")
            st.write("- Content-Based Filtering focus on item features and provide recommendation for users based on users's content relevant.")
            st.write("""- The Content-Based seem to be challenger than The Colaborative-Based due to its learning method.
            Because it learn from the historical data of users, so its hypothesis is if a user was interested in an item in the past, may be they will interested the same thing again in the future.
- And the recommendation will be renew onlyif the data have increased overtime.""")
        st.image("model_info.png")
if __name__ == '__main__':
    main()

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 80px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="white",
        text_align="center",
        height="auto",
        opacity=1
    )

    style_hr = styles(
        display="block",
        #margin=px(2, 2, "auto", "auto"),
        #border_style="inset",
        #border_width=px(1)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer():
    myargs = [
        "Made in ",
        image('https://avatars3.githubusercontent.com/u/45109972?s=400&v=4',
              width=px(25), height=px(25)),
        " with ❤️ by ", "HuynhVinhThang & NguyenVietTruong",
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer()
