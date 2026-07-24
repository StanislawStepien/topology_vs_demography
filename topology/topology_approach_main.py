# Imports #

# Our own dependencies
import cogsNet as cN
import build_network as bn
# Other imports
import os
import sys
import pickle
from datetime import datetime
import pandas as pd
import networkx as nx
import numpy as np
import combu  # MIT License Copyright (c) 2020 Takeru Saito
import matplotlib
from matplotlib import pyplot as plt
from google.cloud import bigquery
# SciKit Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation


print("Begining the topology approach ..")

semesters = range(1, 7)
survey_avg_dates = {  # 'S1': '2011-08-18 22:24:54+0000',
    'S2': '2012-02-08 08:27:26+0000',
    'S3': '2012-05-16 16:25:57+0000',
    'S4': '2012-08-28 16:42:54+0000',
    'S5': '2013-01-26 14:02:27+0000',
    'S6': '2013-05-10 06:33:23+0000'}
for key, item in survey_avg_dates.items():
    survey_avg_dates[key] = datetime.strptime(item, '%Y-%m-%d %H:%M:%S+%f')

best_coding_f1 = {
    "euthanasia": "euthanasia-0.2-iter1.csv",
    "fssocsec": "fssocsec-0.5-iter9.csv",
    "fswelfare": "fswelfare-0.4-iter1.csv",
    "jobguar": "jobguar-0.2-iter8.csv",
    "marijuana": "marijuana-0.1-iter7.csv",
    "toomucheqrights": "toomucheqrights-0.3-iter5.csv"
}

all_records_df = pd.read_csv(
    r'../data/behavioralAll/BehavioralAll_S6_sinceStart_T.csv')
all_records_df['DateTime'] = pd.to_datetime(all_records_df['DateTime'], format='%Y-%m-%d %H:%M:%S+%f:00')

# Any IDs with non-5-digit values are going to be removed - they weren't experiment participants
dict_of_behavioral_df = {}
for key, item in survey_avg_dates.items():
    curr_df = all_records_df
    if key != 'S6':
        curr_df = all_records_df[all_records_df['DateTime'] < item]
    curr_df = curr_df[curr_df.SenderID <= 99999]
    curr_df = curr_df[curr_df.SenderID >= 10000]
    curr_df = curr_df[curr_df.ReceiverID <= 99999]
    curr_df = curr_df[curr_df.ReceiverID >= 10000]
    dict_of_behavioral_df[key] = curr_df


def run_query_raw_out(query):
    # Loading an end-user API key
    CREDS = r'../data/Api_endUserKey.json'
    bigquery.LoadJobConfig(create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
                           write_disposition=bigquery.WriteDisposition.WRITE_EMPTY)
    client = bigquery.Client.from_service_account_json(json_credentials_path=CREDS)
    return client.query_and_wait(query).to_dataframe()


questions = [key for key in best_coding_f1.keys()]
try:
    print("Loading data for semesters...")
    dict_of_dem_surveys_semester = pickle.load(open(
        r"../data/query_results\dict_of_dem_surveys_semester.pkl",
        "rb"))

except FileNotFoundError:
    print(f"Could not find the data. Quering the database..")
    dict_of_dem_surveys_semester = {}
    for semester in list(survey_avg_dates.keys()):
        sem_nr = int(semester[1])
        ds_id_query = f"""SELECT * FROM `NetSense.DemSurvey{semester}`AS DS"""
        dict_of_dem_surveys_semester[semester] = run_query_raw_out(ds_id_query)
        dict_of_dem_surveys_semester[semester] = dict_of_dem_surveys_semester[semester][
            [q + '_' + str(sem_nr) for q in questions]]
        sem_df = dict_of_dem_surveys_semester[semester]
        dict_of_dem_surveys_semester[semester] = dict_of_dem_surveys_semester[semester].dropna(
            subset=[q + '_' + str(sem_nr) for q in questions], axis='index', how='all')
    print(f'Smester data obtained! Saving to ../data/query_results/dict_of_dem_surveys_semester.pkl')
    pickle.dump(dict_of_dem_surveys_semester,
            open(f"../data/query_results/dict_of_dem_surveys_semester.pkl","wb"))

# This recalculates the NetSense values
# "Semester name": [Cij, Tij, Wij, all_nodes, weights_adjacency_matrix]
rerun = False
try:
    if rerun:
        raise FileNotFoundError
    print("Loading CogNet values from the saved file.. "
          "(if you want to calculate them anew - change parameter rerun to True or delete the ../topology_based_appraoch/data/cogsnet_dict.pkl file")
    with open(r'../data/dicts/cogsnet_dict.pkl', "rb") as handle:
        dict_of_cogsnet = pickle.load(handle)
except FileNotFoundError:
    dict_of_cogsnet = {}
    for semesters, BA_df in dict_of_behavioral_df.items():
        print('Running cognet for Semester:', semesters)
        Cij, Tij, Wij, all_nodes, weights_adjacency_matrix = cN.run_cogsnet(BA_df)
        dict_of_cogsnet[semesters] = [Cij, Tij, Wij, all_nodes, weights_adjacency_matrix]
        print('Done for Semester:', semesters)
    with open('../data/dicts/cogsnet_dict.pkl', 'wb') as handle:
        pickle.dump(dict_of_cogsnet, handle)
    with open('../results/cogsnet_dict.pkl', 'wb') as handle:
        pickle.dump(dict_of_cogsnet, handle)

# Building the social networks for each of the semesters
dict_of_networks = {}
for semester, BA_df in dict_of_behavioral_df.items():
    print("Building network for Semester:", semester)
    G: nx.classes.graph.Graph = bn.build_network(BA_df, directed=False)
    dict_of_networks[semester] = G
    print(f'Network for semester {semester} has been built successfully.')


# Making a plot for the networks
plt.figure(figsize=(16, 10))
iterator = 1
val = 230
for key, item in dict_of_networks.items():
    plt.subplot(val + iterator)
    iterator += 1
    node_color = [float(item.degree(v)) for v in item]
    node_size = [float(item.degree(v)) * 5 for v in item]
    nx.draw(item, pos=nx.forceatlas2_layout(item), node_color=node_color, node_size=node_size, cmap='RdBu')
    plt.title(key)
    plt.grid(True)
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.savefig(r"../results/network.png")

# Adding the minoritiy statuses to the networks
network = dict_of_networks['S6']
seed = 9876

# Loading a file where agents are assigned their minority/majority status
minorities_dict_of_dfs = pickle.load(open(r'../data/dicts/dictionary_of_dfs_with_minorities.pkl', 'rb'))
del minorities_dict_of_dfs['gaymarriage_6']
plt.figure(figsize=(16, 10))
iterator = 1
val = 330
titles = ['Gender', 'Ethnicity', 'Privacy setting Facebook', 'English native', 'Parents income', 'Parents education',
          'Parents religion']
for key, item in minorities_dict_of_dfs.items():
    color_map = []
    plt.subplot(val + iterator)
    iterator += 1
    if key == 'momed_1, daded_1':
        print(item)
    for node in network:
        if node in item.values:
            color_map.append('blue')
        else:
            color_map.append('orange')
    nx.draw(network, pos=nx.forceatlas2_layout(network, seed=seed), node_color=color_map, node_size=50)
    plt.title(titles[iterator - 2])
    plt.grid(True)
plt.subplot(338)
node_color = [float(network.degree(v)) for v in network]
node_size = [float(network.degree(v)) * 5 for v in network]
nx.draw(network, pos=nx.forceatlas2_layout(network, seed=seed), node_color=node_color, node_size=node_size, cmap='RdBu')
plt.title('Node degree')
plt.grid(True)

plt.subplot(339)
pr = [x * 5000 for x in nx.pagerank(network).values()]
nx.draw(network, pos=nx.forceatlas2_layout(network, seed=seed), node_color=pr, node_size=pr, cmap='RdBu')
plt.title('Pagerank')
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)


# Preparing the training features
def get_features(network: nx.classes.graph.Graph, matrix_of_number_of_interactions: np.ndarray,
                 matrix_of_cogsnet_weights: np.ndarray, matrix_of_dates_of_last_interaction: np.ndarray,
                 list_of_all_nodes: list[int]) -> dict[str:list]:
    # First part of this foo calculates the 'personal' measures for each of the agents
    (degrees, cogsnetSum, interactionSum, avgTimeSinceLastInteraction, centrality_degree_list,
     centrality_betweenness_list, pagerank_list, eigenvactor_centrality_list, closeness_centrality_list,
     current_flow_closeness_centrality_list, information_centrality_list, load_list, subgraph_centrality_exp_list,
     laplacian_list) = [], [], [], [], [], [], [], [], [], [], [], [], [], []

    centrality_degree = nx.degree_centrality(network)
    centrality_betweenness = nx.betweenness_centrality(network, k=len(list_of_all_nodes) - 1)
    pagerank = nx.pagerank(network)
    eigenvactor_centrality = nx.eigenvector_centrality(network)
    closeness_centrality = nx.closeness_centrality(network)
    current_flow_closeness_centrality = nx.current_flow_closeness_centrality(network)
    information_centrality = nx.information_centrality(network)
    load = nx.load_centrality(network)
    subgraph_centrality_exp = nx.subgraph_centrality_exp(network)
    laplacian = nx.laplacian_centrality(network)

    for node in list_of_all_nodes:
        degrees.append(network.degree(node))
        curr_index = list_of_all_nodes.index(node)
        cogsnetSum.append(sum(matrix_of_cogsnet_weights[curr_index]))
        interactionSum.append(sum(matrix_of_number_of_interactions[curr_index]))
        centrality_degree_list.append(centrality_degree[node])
        centrality_betweenness_list.append(centrality_betweenness[node])
        pagerank_list.append(pagerank[node])
        eigenvactor_centrality_list.append(eigenvactor_centrality[node])
        closeness_centrality_list.append(closeness_centrality[node])
        current_flow_closeness_centrality_list.append(current_flow_closeness_centrality[node])
        information_centrality_list.append(information_centrality[node])
        load_list.append(load[node])
        subgraph_centrality_exp_list.append(subgraph_centrality_exp[node])
        laplacian_list.append(laplacian[node])
        # avgTimeSinceLastInteraction is not going to be used as a feature,
        # because it's not really topological in nature. I'm leaving it here for now, but it's not getting used later
        tsum = 0
        for register in matrix_of_dates_of_last_interaction[curr_index]:
            if register != 0:
                register = register.timestamp()
                timeFromNow = pd.Timestamp.now().timestamp() - register
                tsum += timeFromNow
        avgTimeSinceLastInteraction.append(tsum / len(matrix_of_dates_of_last_interaction[curr_index]))
    # Second part of this foo calculates the aggregate measures for each agents' neighbours
    avgNeighboursCogsNetSums, avgNeighboursInteractionSums, avgNeighbourDegrees = [], [], []
    for node in list_of_all_nodes:
        neighbours = network.neighbors(node)
        cogsnet = 0
        interaction = 0
        degree = 0
        # Calculating first the sum of parameters for each of the node's neighbors
        for neighbour in neighbours:
            cogsnet += sum(matrix_of_cogsnet_weights[list_of_all_nodes.index(neighbour)])
            interaction += sum(matrix_of_number_of_interactions[list_of_all_nodes.index(neighbour)])
            degree += network.degree(neighbour)
        # Getting the average of each parameter 'per neighbor' of the node
        nodes_degree = degrees[list_of_all_nodes.index(node)]
        # noinspection PyTypeChecker
        avgNeighboursCogsNetSums.append(cogsnet / nodes_degree)
        # noinspection PyTypeChecker
        avgNeighboursInteractionSums.append(interaction / nodes_degree)
        # noinspection PyTypeChecker
        avgNeighbourDegrees.append(degree / nodes_degree)

    # Parsing data into a dict of lists,
    # where n-th entry in every list corresponds to the features_dict['EgoID'][n]'s agent.
    features_dictionary = {"EgoID": list_of_all_nodes, "Node degree": degrees, "Sum of CogsNet": cogsnetSum,
                           "Avg Neighbour's CogsNetSum": avgNeighboursCogsNetSums,
                           "Avg Neighbour Degree": avgNeighbourDegrees,
                           "Avg Neighbour's InteractionSum": avgNeighboursInteractionSums,
                           "Degree-centrality": centrality_degree_list,
                           "Betweenness-centrality": centrality_betweenness_list,
                           "Pagerank": pagerank_list,
                           "Eigenvector centrality": eigenvactor_centrality_list,
                           "Closeness centrality": closeness_centrality_list,
                           "Current flow closeness centrality": current_flow_closeness_centrality_list,
                           "Information centrality": information_centrality_list,
                           "Load": load_list,
                           "Subgraph centrality exp": subgraph_centrality_exp_list,
                           "Laplacian": laplacian_list}
    return features_dictionary


try:
    print("Trying to read dict of training data..")
    dict_of_training_dfs = pickle.load(open('../data/dicts/dict_of_training_dfs.pkl', "rb"))
except FileNotFoundError:
    print("failed, now regenerating it..")
    dict_of_training_dfs = {}
    for sem in list(survey_avg_dates.keys()):
        features_dict = get_features(network=dict_of_networks[sem],
                                     matrix_of_number_of_interactions=dict_of_cogsnet[sem][0],
                                     matrix_of_cogsnet_weights=dict_of_cogsnet[sem][2],
                                     matrix_of_dates_of_last_interaction=dict_of_cogsnet[sem][1],
                                     list_of_all_nodes=dict_of_cogsnet[sem][3])
        features_df = pd.DataFrame.from_dict(features_dict)
        features_df['Semester'] = int(sem[1])
        dict_of_training_dfs[sem] = features_df
    pickle.dump(dict_of_training_dfs, open(r'../data/dicts/dict_of_training_dfs.pkl', "wb"))

questions = ["euthanasia", "fssocsec", "fswelfare", "jobguar", "marijuana", "toomucheqrights"]
coding_dir = '../data/CoDING_results/'
dict_of_coding_best_res = {}
for sem in list(survey_avg_dates.keys()):
    dict_of_coding_best_res[sem] = {}
    semester = int(sem[1])
    for question in questions:
        coding_df_q = pd.read_csv(coding_dir + best_coding_f1[question], sep=';')
        semester_df = coding_df_q[coding_df_q.SurveyNr == semester]
        semester_df = semester_df[['StudentID', 'SurveyNr', 'Question', 'OpinionSurvey', 'OpinionSim']]
        semester_df['coding_success'] = np.where(semester_df['OpinionSim'] == semester_df['OpinionSurvey'], 1, 0)
        semester_df = semester_df[['StudentID', 'SurveyNr', 'Question', 'coding_success']]
        dict_of_coding_best_res[sem][question] = semester_df
# merging features into a single place
features = ["Node degree", "Sum of CogsNet", "Avg Neighbour's CogsNetSum", "Avg Neighbour Degree", "Degree-centrality",
            "Betweenness-centrality", "Pagerank", "Eigenvector centrality", "Closeness centrality",
            "Current flow closeness centrality", "Load", "Subgraph centrality exp",
            "Laplacian"]
list_of_df_same_semester = []
for semester in list(survey_avg_dates.keys()):
    list_of_df_questions_same_sem = []
    for question in questions:
        left = dict_of_training_dfs[semester]
        right = dict_of_coding_best_res[semester][question][['StudentID', 'SurveyNr', 'Question', 'coding_success']]
        temp_df = pd.merge(left, right, left_on='EgoID', right_on='StudentID')
        list_of_df_questions_same_sem.append(temp_df)
    question_df = pd.concat(list_of_df_questions_same_sem)
    list_of_df_same_semester.append(question_df)
all_sem_all_q_df = pd.concat(list_of_df_same_semester)
all_sem_all_q_df = all_sem_all_q_df[['EgoID', 'SurveyNr', 'Question', 'coding_success'] + features]
question_val = {}
val = 1
for question in questions:
    question_val[question] = val
    val += 1
for question in questions:
    all_sem_all_q_df.loc[all_sem_all_q_df["Question"] == question, "Question"] = str(question_val[question])
all_sem_all_q_df = all_sem_all_q_df.astype({'Question': int})
all_sem_all_q_df.to_csv(r'../data/csv/topology_based_features.csv')
all_sem_all_q_df.to_csv(r'../results/topology_based_features.csv')

# Splitting data before training
from sklearn.model_selection import train_test_split

dataX = all_sem_all_q_df[
    features + ['SurveyNr', 'Question', 'EgoID']]  # Survey is only for the histogram below, it's removed later

dataX = dataX.T.drop_duplicates().T
# features.remove("Information centrality")
dataY = all_sem_all_q_df[["coding_success"]]
train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

# train is now 70% of the entire data set
x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio, random_state=123)

# test is now 15% of the initial data set
# validation is now 15% of the initial data set
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio),
                                                random_state=123)

# Don't wanna have survey nr in my training data
x_train = x_train.drop(columns=['SurveyNr'])
x_val = x_val.drop(columns=['SurveyNr'])
x_test = x_test.drop(columns=['SurveyNr'])

classifiers = {
    # LogisticRegression: {"penalty": ["l1", "l2", 'elasticnet', None], "tol": [0.01, 0.001, 0.0001, 0.00001],
    # "fit_intercept": [True, False], "class_weight": ['balanced', None],
    # "solver": ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']},
    DecisionTreeClassifier: {"max_depth": [3, 4, 5, 6, 7, 10], "max_features": ["sqrt", "log2", None],
                             "min_samples_split": [2, 5], "min_samples_leaf": [1, 3, 5, 7, 12, 20],
                             "max_leaf_nodes": [2, 6, 12, 20], "criterion": ["gini", "entropy", "log_loss"],
                             "splitter": ["best", "random"]},
    #

    # RandomForestClassifier: {"n_estimators": [50, 100], "criterion": ["gini", "entropy", "log_loss"],
    #                        "max_depth": [5, 25, 50], "min_samples_leaf": [1, 5, 10],
    #                       "max_features": ["sqrt", "log2", None], "bootstrap": [True, False]},
    # ExtraTreeClassifier: {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"],
    #                      "max_depth": [5, 10, 50], "max_features": ["sqrt", "log2", None],
    #                     "min_samples_leaf": [1, 5, 10]},
    # GaussianNB: {},
    # KNeighborsClassifier: {"n_neighbors": [1, 5, 10, 15, 20], "weights": ["uniform", "distance"],
    #                      "algorithm": ["auto", "ball_tree", "kd_tree", "brute"], "p": [1, 3, 5]}
}


def repeat_fitting_n_times_get_acc_and_f1(number_of_runs: int, accuracy: float, f1_score_value: float, X_set, Y_set,
                                          model_temp_result: DecisionTreeClassifier or ExtraTreeClassifier
                                                             or RandomForestClassifier):
    X_train_set, X_val_set, Y_train_set, Y_val_set = train_test_split(X_set, Y_set, test_size=0.3)
    for _ in range(number_of_runs):
        if model_temp_result.__class__ == RandomForestClassifier:
            model_temp_result = model_temp_result.fit(X_train_set, Y_train_set.values.ravel())
        else:
            model_temp_result = model_temp_result.fit(X_train_set, Y_train_set)
        # Predict the response for test dataset
        Y_pred_set = model_temp_result.predict(X_val_set)
        # Model Accuracy, how often is the classifier correct?
        accuracy += metrics.accuracy_score(Y_val_set, Y_pred_set)
        f1_score_value += metrics.f1_score(Y_val_set, Y_pred_set)
    accuracy /= number_of_runs
    f1_score_value /= number_of_runs
    return accuracy, f1_score_value, model_temp_result


def get_model_f1_acc_from_n_runs(model, n, X_data_df, Y_data_df):
    f1_measure_score = 0
    accuracy_score = 0
    tn, fp, fn, tp = 0, 0, 0, 0
    for i in range(n):
        Y_predicted = model.predict(X_data_df)
        temp_f1 = metrics.f1_score(Y_data_df, Y_predicted)
        acc_temp = metrics.accuracy_score(Y_data_df, Y_predicted)
        ttn, tfp, tfn, ttp = confusion_matrix(Y_data_df, Y_predicted).ravel()
        f1_measure_score += temp_f1
        accuracy_score += acc_temp
        tn += ttn
        fp += tfp
        fn += tfn
        tp += ttp
    f1_measure_score /= n
    accuracy_score /= n
    tn /= n
    fp /= n
    fn /= n
    tp /= n
    # print(f"""{tn}\t|{fp}\n_\t_\n{fn}\t|{tp}""")
    return f1_measure_score, accuracy_score, [tn, fp, fn, tp]


results = {'model': [], 'params': [], "accuracy": [], "f1_score": []}
best_accuracy: float = 0.0
best_f1_score: float = 0.0
# Merging sets back again (except for the Test set) so that program re-splits them for every iteration anew
X = pd.concat([x_train, x_val])
Y = pd.concat([y_train, y_val])
X = X.T.drop_duplicates().T
X = X.drop(columns=['EgoID'])

# no of times splitting of the dataset, fitting of the model and evaluating its performance will be done
n = 1
best_model = None
for model, classifier_attributes in classifiers.items():
    # combu helps iterate over all combinations of the models' parameters
    model_parametrizer = combu.Combu(model, progress=True)
    for test_model, parameters in model_parametrizer.execute(classifier_attributes):
        acc = 0
        f1_score = 0
        acc, f1_score, temp_model = repeat_fitting_n_times_get_acc_and_f1(n, acc, f1_score, X, Y, test_model)
        results['model'].append(model)
        results['params'].append(parameters)
        results['accuracy'].append(acc)
        results['f1_score'].append(f1_score)
        if acc > best_accuracy:
            best_accuracy = acc
        if f1_score >= best_f1_score:
            # if curr f1 score is identical as the previous one, but has greater acc, then it will be taken as the new best.
            if f1_score == best_f1_score and acc >= best_accuracy:
                best_f1_score = f1_score
                best_model = temp_model
                best_params = parameters
            else:
                best_f1_score = f1_score
                best_accuracy = acc
                best_model = temp_model
                best_params = parameters
            print(f"The best model was: {best_model} "
                  f"of parameters:{best_params}, with accuracy {best_accuracy} and f1 score {best_f1_score}")
with open(r"/data/best_models/Model_high.pkl", 'wb') as filename:
    # noinspection PyTypeChecker
    pickle.dump(best_model, filename)
with open(r"../results/Model_high.pkl", 'wb') as filename:
    # noinspection PyTypeChecker
    pickle.dump(best_model, filename)

features.append("Question")

x_for_pred_test = x_test[features]
x_for_pred_test = x_for_pred_test.T.drop_duplicates().T
"""print(x_test)

model = pickle.load(open('/code/topology_based_approach/The_best_model_S_All.pkl', 'rb'))
y_pred = model.predict(x_for_pred_test)
print(metrics.f1_score(y_pred, y_test), metrics.accuracy_score(y_pred, y_test))"""
model_file = r"../data/best_models/The_best_model_S_All.pkl"
model = pickle.load(open(model_file, 'rb'))
y_pred = model.predict(x_for_pred_test)

tree.plot_tree(model, feature_names=features)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8, 3)
fig.savefig('../results/tree.png')
minorities_dict_of_dfs = pickle.load(open(r'../data/dicts/dictionary_of_dfs_with_minorities.pkl', 'rb'))
del minorities_dict_of_dfs['gaymarriage_6']
minority_results_dict = {}
X = pd.concat([x_train, x_val])
Y = pd.concat([y_train, y_val])
data = pd.concat([X, Y], axis=1)

model = pickle.load(open(model_file, 'rb'))
for minority, minority_df_of_ids in minorities_dict_of_dfs.items():
    print('Experiment for minority: ', minority)
    # print("len of minority df", len(minority_df_of_ids))
    min_res = data[data["EgoID"].isin(minority_df_of_ids)]
    X = min_res[features]
    Y = min_res['coding_success']
    X = X.T.drop_duplicates().T
    # print("Len of X", len(X))
    # print("Len of Y", len(Y))
    f1_min, acc_min, conf_mat = get_model_f1_acc_from_n_runs(model, 1, X, Y)
    minority_results_dict[minority] = f1_min

X = pd.concat([x_train, x_val])
Y = pd.concat([y_train, y_val])
data = pd.concat([X, Y], axis=1)
for question in questions:
    q_data = data.loc[data['Question'] == question_val[question]]
    # print(q_data.mean(axis=0))
    q_X = q_data[features]
    q_X = q_X.T.drop_duplicates().T
    q_Y = q_data[['coding_success']]
    # print("len of question df", len(q_X))
    acc, f1_score, conf_mat = get_model_f1_acc_from_n_runs(model, 1, q_X, q_Y)
    #disp = ConfusionMatrixDisplay(confusion_matrix=*conf_mat, display_labels=model.classes_)
    #disp.plot()
    # plt.show()

data = pd.concat([x_test, y_test], axis=1)
unseen_data_minority_f1 = {}
for minority, minority_df_of_ids in minorities_dict_of_dfs.items():
    print('\n\nExperiment for minority on unseen data: ', minority)
    print("total len of minority:", len(minority_df_of_ids))
    min_res = data[data["EgoID"].isin(minority_df_of_ids)]
    print(min_res.std(axis=0))
    for res in list(min_res.std(axis=0)):
        print(res)
    print('Number of minority entries in the unseen data:', len(min_res))
    X = min_res[features]
    X = X.T.drop_duplicates().T
    Y = min_res['coding_success']
    f1_minority_unseen, acc_minority_unseen, conf_mat = get_model_f1_acc_from_n_runs(model, 1, X, Y)
    print(f"F1 score: {f1_minority_unseen}\nAccuracy score: {acc_minority_unseen}")
    unseen_data_minority_f1[minority] = f1_minority_unseen

X = pd.concat([x_train, x_val])
Y = pd.concat([y_train, y_val])
data = pd.concat([X, Y], axis=1)
# for res in list(data.std(axis=0)):
#   print(res)
minority_results_dict = {}
for question in questions:
    minority_results_dict[question] = []
    print('\n#################Experiment for question: ', question)
    q_data = data.loc[data['Question'] == question_val[question]]
    for minority, minority_df_of_ids in minorities_dict_of_dfs.items():
        print("len of minority df", len(minority_df_of_ids))
        min_res = q_data[q_data["EgoID"].isin(minority_df_of_ids)]
        print('\nExperiment for minority: ', minority, 'members:', len(min_res))
        X = min_res[features]
        X = X.T.drop_duplicates().T
        for res in list(X.std(axis=0)):
            print(res)
        Y = min_res['coding_success']
        # print("Len of X", len(X))
        # print("Len of Y", len(Y))
        f1_min, acc_min, conf_mat = get_model_f1_acc_from_n_runs(model, 1, X, Y)
        print(f"F1 score: {f1_min}\nAccuracy score: {acc_min}")
        minority_results_dict[question].append(f1_min)
print(minority_results_dict)
result_df = pd.DataFrame.from_dict(minority_results_dict)
print(f"F1_{minority}:{f1_min},",end='')


data = pd.concat([x_test, y_test], axis=1)
minority_results_dict = {}
min_acc_dict = {}
index_labels = ['All', "gender", "ethnicity", "fb_privacy", "en_native", "p_income", "p_ed", "p_rel"]
y_pred_all = model.predict(x_test[features])
all_f1 = metrics.f1_score(y_test, y_pred_all)

for question in questions:
    minority_results_dict[question] = []
    min_acc_dict[question] = []
    q_data = data.loc[data['Question'] == question_val[question]]
    X = q_data[features]
    X = X.T.drop_duplicates().T
    Y = q_data['coding_success']
    f1_all, acc_all, conf_mat = get_model_f1_acc_from_n_runs(model, 10, X, Y)
    minority_results_dict[question].append(f1_all)
    for minority, minority_df_of_ids in minorities_dict_of_dfs.items():
        # print("len of minority df", len(minority_df_of_ids))
        min_res = q_data[q_data["EgoID"].isin(minority_df_of_ids)]
        # print('\nExperiment for minority: ', minority, 'members:', len(min_res))
        X = min_res[features]
        X = X.T.drop_duplicates().T
        Y = min_res['coding_success']
        # print("Len of X", len(X))
        # print("Len of Y", len(Y))
        f1_min, acc_min, conf_mat = get_model_f1_acc_from_n_runs(model, 10, X, Y)
        minority_results_dict[question].append(f1_min)
        min_acc_dict[question].append(acc_min)

result_df = pd.DataFrame(minority_results_dict, index=index_labels)
print(result_df)
# print(f"F1_{minority}:{f1_min},",end='')

print("Ended the topology approach.")

