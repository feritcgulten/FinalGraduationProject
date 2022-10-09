from flask import Flask, render_template, redirect, send_file
from flask.helpers import url_for
from flask_bootstrap import Bootstrap
from sqlalchemy import NVARCHAR
from dashboard import create_dashboard
from dashboard import dashboards
#from application.dashboard import app
from flask_login import LoginManager, login_user
from flask_login.mixins import UserMixin
import numpy as np
import pandas as pd
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import sys

app = Flask(__name__)
app.config["SECRET_KEY"] = "THIS IS A SECRET, DON'T DO THIS!"
Bootstrap(app)
create_dashboard(app)
dashboards(app)


@app.route("/influencers/")
def influencers():
    return render_template('influence.html',title="Influencers")

@app.route("/")
def home():
    return render_template('home.html',title="Home")

df = pd.read_json (r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\query_hastane-asi-randevu_since_2020-01-01_tweets_18-08-47 22-03-2022.json')    
in_reply_to_user_id= df.loc[ :, "in_reply_to_user_id" ]
result_one = in_reply_to_user_id

user_id= df.loc[ :, "user_id"]
result_two = user_id

nodes= np.concatenate((result_one, result_two))
nodes = list( dict.fromkeys(nodes) )
nodes = [x for x in nodes if math.isnan(x) == False]

new_df = pd.DataFrame(columns = ['in_reply_to_user_id', 'user_id'])
new_df['in_reply_to_user_id'] = in_reply_to_user_id
new_df['user_id'] = user_id

# remove rows by filtering
new_df = new_df.apply (pd.to_numeric, errors='coerce')
new_df = new_df.dropna()
# display the dataframe
combined = new_df.values.tolist()    

    
t_131 = nx.DiGraph()
t_131.add_nodes_from(nodes)
t_131.add_edges_from(combined)
    
betCent = nx.betweenness_centrality(t_131, normalized=True, endpoints=True)
eigenCent = nx.eigenvector_centrality(t_131,max_iter=1000)
degCent= nx.degree_centrality(t_131)
page= nx.pagerank(t_131, alpha = 0.8) 
closeCent=nx.closeness_centrality(t_131)

nodes_bet =sorted(betCent, key=betCent.get, reverse=True)[:10] #top five nodes
nodes_eig =sorted(eigenCent, key=eigenCent.get, reverse=True)[:10] #top five nodes
nodes_deg =sorted(degCent, key=degCent.get, reverse=True)[:10]  #top five nodes
nodes_page =sorted(page, key=page.get, reverse=True)[:10]  #top five nodes
nodes_close=sorted(closeCent, key=closeCent.get, reverse=True)[:10]  #top five nodes

@app.route('/between_cent/')
def test():
    labels={}
    for node in t_131.nodes():
        if node in nodes_bet:
        #set the node name as the key and the label as its value 
            labels[node] = node

    # plotting considering centrality
    pos = nx.spring_layout(t_131)
    node_color = [500000.0 * t_131.degree(v) for v in t_131]
    node_size =  [v * 6000000 for v in betCent.values()]
    plt.figure(figsize=(8,8))
    plt.title('Betweennes Centrality')
    nx.draw(t_131,node_color=node_color,
    node_size=0, with_labels=False, edge_color ='white',linewidths=0,width=0)
    nx.draw_networkx(t_131, pos=pos,labels=labels,font_size=10,font_color='blue',style='dashed',
                  node_color=node_color,
                  node_size=node_size,
                  width=0.2,font_weight='bold')
    nx.draw_networkx_edges(t_131,pos=pos,alpha=0.3,width=0.2,)
    plt.axis('off');
    
    img= BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image_bet.png')


df_2 = pd.read_json (r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\query_hastane-entube_since_2020-01-01_tweets_18-04-39 22-03-2022.json')    
in_reply_to_user_id_2= df_2.loc[ :, "in_reply_to_user_id" ]
result_one_2 = in_reply_to_user_id_2

user_id_2= df_2.loc[ :, "user_id"]
result_two_2 = user_id_2

nodes_2= np.concatenate((result_one_2, result_two_2))
nodes_2 = list( dict.fromkeys(nodes_2) )
nodes_2 = [x for x in nodes_2 if math.isnan(x) == False]

new_df_2 = pd.DataFrame(columns = ['in_reply_to_user_id', 'user_id'])
new_df_2['in_reply_to_user_id'] = in_reply_to_user_id_2
new_df_2['user_id'] = user_id_2

# remove rows by filtering
new_df_2 = new_df_2.apply (pd.to_numeric, errors='coerce')
new_df_2 = new_df_2.dropna()
# display the dataframe
combined_2 = new_df_2.values.tolist()    

    
t_131_2 = nx.DiGraph()
t_131_2.add_nodes_from(nodes_2)
t_131_2.add_edges_from(combined_2)

betCent_2 = nx.betweenness_centrality(t_131_2, normalized=True, endpoints=True)
nodes_bet_2 =sorted(betCent_2, key=betCent_2.get, reverse=True)[:10] #top five nodes



@app.route('/between_cent_2/')
def btw_2():
    labels={}
    for node in t_131_2.nodes():
        if node in nodes_bet_2:
        #set the node name as the key and the label as its value 
            labels[node] = node
# plotting considering centrality
    
    pos = nx.spring_layout(t_131_2)
    node_color = [500000.0 * t_131_2.degree(v) for v in t_131_2]
    node_size =  [v * 6000000 for v in betCent_2.values()]
    plt.figure(figsize=(8,8))
    plt.title('Betweennes Centrality')
    nx.draw(t_131_2,node_color=node_color,
    node_size=0, with_labels=False, edge_color ='white',linewidths=0,width=0)
    nx.draw_networkx(t_131_2, pos=pos,labels=labels,font_size=10,font_color='blue',style='dashed',
                  node_color=node_color,
                  node_size=node_size,
                  width=0.2,font_weight='bold')
    nx.draw_networkx_edges(t_131_2,pos=pos,alpha=0.3,width=0.2,)
    plt.axis('off');
    
    img= BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image_bet.png')

df_3 = pd.read_json (r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\query_hastane-pcr_since_2020-01-01_tweets_18-02-43 22-03-2022.json')    
in_reply_to_user_id_3= df_3.loc[ :, "in_reply_to_user_id" ]
result_one_3 = in_reply_to_user_id_3

user_id_3= df_3.loc[ :, "user_id"]
result_two_3 = user_id_3

nodes_3= np.concatenate((result_one_3, result_two_3))
nodes_3 = list( dict.fromkeys(nodes_3) )
nodes_3 = [x for x in nodes_3 if math.isnan(x) == False]

new_df_3 = pd.DataFrame(columns = ['in_reply_to_user_id', 'user_id'])
new_df_3['in_reply_to_user_id'] = in_reply_to_user_id_3
new_df_3['user_id'] = user_id_3

# remove rows by filtering
new_df_3 = new_df_3.apply (pd.to_numeric, errors='coerce')
new_df_3 = new_df_3.dropna()
# display the dataframe
combined_3 = new_df_3.values.tolist()    

    
t_131_3 = nx.DiGraph()
t_131_3.add_nodes_from(nodes_3)
t_131_3.add_edges_from(combined_3)

betCent_3 = nx.betweenness_centrality(t_131_3, normalized=True, endpoints=True)
nodes_bet_3 =sorted(betCent_3, key=betCent_3.get, reverse=True)[:10] #top five nodes



@app.route('/between_cent_3/')
def btw_3():
    labels={}
    for node in t_131_3.nodes():
        if node in nodes_bet_3:
        #set the node name as the key and the label as its value 
            labels[node] = node
# plotting considering centrality
    pos = nx.spring_layout(t_131_3)
    node_color = [500000.0 * t_131_3.degree(v) for v in t_131_3]
    node_size =  [v * 6000000 for v in betCent_3.values()]
    plt.figure(figsize=(8,8))
    plt.title('Betweennes Centrality')
    nx.draw(t_131_3,node_color=node_color,
    node_size=0, with_labels=False, edge_color ='white',linewidths=0,width=0)
    nx.draw_networkx(t_131_3, pos=pos,labels=labels,font_size=10,font_color='blue',style='dashed',
                  node_color=node_color,
                  node_size=node_size,
                  width=0.2,font_weight='bold')
    nx.draw_networkx_edges(t_131_3,pos=pos,alpha=0.3,width=0.2,)
    plt.axis('off');
    
    img= BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image_bet_3.png')



df_4 = pd.read_json (r'C:\Users\ekol1\Desktop\Ceren\GraduationProject\saglik\saglik\query_yogun-bakimda-yer_since_2020-01-01_tweets_17-37-33 22-03-2022.json')    
in_reply_to_user_id_4= df_4.loc[ :, "in_reply_to_user_id" ]
result_one_4= in_reply_to_user_id_4

user_id_4= df_4.loc[ :, "user_id"]
result_two_4 = user_id_4

nodes_4= np.concatenate((result_one_4, result_two_4))
nodes_4 = list( dict.fromkeys(nodes_4) )
nodes_4 = [x for x in nodes_4 if math.isnan(x) == False]

new_df_4 = pd.DataFrame(columns = ['in_reply_to_user_id', 'user_id'])
new_df_4['in_reply_to_user_id'] = in_reply_to_user_id_4
new_df_4['user_id'] = user_id_4

# remove rows by filtering
new_df_4 = new_df_4.apply (pd.to_numeric, errors='coerce')
new_df_4 = new_df_4.dropna()
# display the dataframe
combined_4 = new_df_4.values.tolist()    

    
t_131_4 = nx.DiGraph()
t_131_4.add_nodes_from(nodes_4)
t_131_4.add_edges_from(combined_4)

betCent_4 = nx.betweenness_centrality(t_131_4, normalized=True, endpoints=True)
nodes_bet_4 =sorted(betCent_4, key=betCent_4.get, reverse=True)[:10] #top five nodes



@app.route('/between_cent_4/')
def btw_4():
    labels={}
    for node in t_131_4.nodes():
        if node in nodes_bet_4:
        #set the node name as the key and the label as its value 
            labels[node] = node
# plotting considering centrality
    

    pos = nx.spring_layout(t_131_4)
    node_color = [500000.0 * t_131_4.degree(v) for v in t_131_4]
    node_size =  [v * 6000000 for v in betCent_4.values()]
    plt.figure(figsize=(8,8))
    plt.title('Betweennes Centrality')
    nx.draw(t_131_4,node_color=node_color,
    node_size=0, with_labels=False, edge_color ='white',linewidths=0,width=0)
    nx.draw_networkx(t_131_4, pos=pos,labels=labels,font_size=10,font_color='blue',style='dashed',
                  node_color=node_color,
                  node_size=node_size,
                  width=0.2,font_weight='bold')
    nx.draw_networkx_edges(t_131_4,pos=pos,alpha=0.3,width=0.2,)
    plt.axis('off');
    
    img= BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image_bet_4.png')


@app.route('/topic4/')
def topic4():
    return   render_template( "topic4.html", nodes_bet_4 = nodes_bet_4, betCent_4= betCent_4)

@app.route('/topic3/')
def topic3():
    return   render_template( "topic3.html", nodes_bet_3 = nodes_bet_3, betCent_3= betCent_3)

@app.route('/topic2/')
def topic2():
    return   render_template( "topic2.html", nodes_bet_2 = nodes_bet_2 , betCent_2= betCent_2)
@app.route('/topic1/')
def pokes():  
    return render_template("topic1.html",  nodes_bet = nodes_bet , betCent= betCent)

if __name__ == "__main__":
    app.run()