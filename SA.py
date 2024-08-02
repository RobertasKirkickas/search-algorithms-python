#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Inline mode
get_ipython().run_line_magic('matplotlib', 'inline')

# Import necessary libraries
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque, namedtuple
from queue import PriorityQueue

# PGMPY - Bayesian Network related imports
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import (
    HillClimbSearch, 
    K2Score, 
    BicScore, 
    PC, 
    TreeSearch, 
    MmhcEstimator, 
    StructureEstimator
)

# Scikit-learn related imports
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression

# NLTK related imports for text processing
import nltk
from nltk.corpus import stopwords
import string
import re

# Download stopwords if not already downloaded
# nltk.download('stopwords')


# Task 1 – Search methods

# In[2]:


Node = namedtuple('Node', ['state', 'parent', 'action', 'path_cost']) # class is used to represent nodes in a graph-tree structure.


# In[3]:


# Defining Graph class
class Graph:
    def __init__(self, structure):
        # Initialize the Graph with a structure dictionary representing the adjacency list
        self.structure = structure
    
    def successors(self, state):
         # Retrieve and return the list of successors for a given state from the structure
        return self.structure.get(state, [])


# In[4]:


structure = {
    'Ipswich': [('Colchester', 'Ipswich -> Colchester', 40), ('Bury St Edmunds', 'Ipswich -> Bury St Edmunds', 50), ('Norwich', 'Ipswich -> Norwich', 70), ('Chelmsford', 'Ipswich -> Chelmsford', 35)],
    'Colchester': [('Chelmsford', 'Colchester -> Chelmsford', 30), ('Norwich', 'Colchester -> Norwich', 45), ('Felixstowe', 'Colchester -> Felixstowe', 60)],
    'Norwich': [('Great Yarmouth', 'Norwich -> Great Yarmouth', 20), ('Lowestoft', 'Norwich -> Lowestoft', 40), ('Cambridge', 'Norwich -> Cambridge', 55), ('Thetford', 'Norwich -> Thetford', 65)],
    'Bury St Edmunds': [('Cambridge', 'Bury St Edmunds -> Cambridge', 60), ('Colchester', 'Bury St Edmunds -> Colchester', 45), ('Stowmarket', 'Bury St Edmunds -> Stowmarket', 40)],
    'Chelmsford': [('London', 'Chelmsford -> London', 40), ('Cambridge', 'Chelmsford -> Cambridge', 50), ('Harwich', 'Chelmsford -> Harwich', 55)],
    'Lowestoft': [('Great Yarmouth', 'Lowestoft -> Great Yarmouth', 25), ('Yarmouth', 'Lowestoft -> Yarmouth', 45)],
    'Great Yarmouth': [('Lowestoft', 'Great Yarmouth -> Lowestoft', 25), ('Norwich', 'Great Yarmouth -> Norwich', 20), ('Caister', 'Great Yarmouth -> Caister', 30)],
    'Cambridge': [('London', 'Cambridge -> London', 50), ('Chelmsford', 'Cambridge -> Chelmsford', 50), ('Ely', 'Cambridge -> Ely', 30)],
    'London': [('Chelmsford', 'London -> Chelmsford', 40), ('Cambridge', 'London -> Cambridge', 55), ('Colchester', 'London -> Colchester', 70)],
    'Felixstowe': [('Colchester', 'Felixstowe -> Colchester', 60)],
    'Thetford': [('Norwich', 'Thetford -> Norwich', 65)],
    'Stowmarket': [('Bury St Edmunds', 'Stowmarket -> Bury St Edmunds', 40)],
    'Harwich': [('Chelmsford', 'Harwich -> Chelmsford', 55)],
    'Yarmouth': [('Lowestoft', 'Yarmouth -> Lowestoft', 45)],
    'Caister': [('Great Yarmouth', 'Caister -> Great Yarmouth', 30)],
    'Ely': [('Cambridge', 'Ely -> Cambridge', 30)]
}

instance = Graph(structure)


# In[5]:


def draw_graph(graph, node_color='skyblue', seed=1):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Iterate through each node and its successors in the graph
    for node, successors in graph.items():
        for successor in successors:
            successor_node, action, cost = successor
            ''' Add an edge from the current node to its successor
                with action and cost attributes '''
            G.add_edge(node, successor_node, action=action, weight=cost)

    # Compute the layout of the graph using spring layout
    pos = nx.spring_layout(G, seed=seed, k=9) 
    
    # Create a new figure
    plt.figure(figsize=(12, 22))
    
    # Draw the graph with nodes labeled
    nx.draw(G, pos, with_labels=True, node_color=node_color, node_size=1000, font_size=10, font_weight='bold',
            edge_color='gray', arrowsize=20, connectionstyle='arc3,rad=0.1')

    # Adding labels (actions and costs) to the edges
    labels = {(u, v): f"{d['action']} ({d['weight']})" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='green')

    # Display the graph
    plt.show()


# In[6]:


draw_graph(structure, seed = 11)


# In[7]:


# Extracting the paths as a string
def extract_path(node):
    path = []
    while node.parent is not None:
        path.insert(0, node.state)
        node = node.parent
    path.insert(0, node.state)
    return " -> ".join(path)


# In[8]:


# Heuristic values
heuristic_structure_value = {
    'Ipswich': 70,
    'Colchester': 40,
    'Norwich': 100,
    'Bury St Edmunds': 50,
    'Chelmsford': 30,
    'Lowestoft': 120,
    'Great Yarmouth': 130,
    'Cambridge': 60,
    'London': 20,
    'Felixstowe': 70,
    'Thetford': 110,
    'Stowmarket': 60,
    'Harwich': 90,
    'Yarmouth': 105,
    'Caister': 120,
    'Ely': 90
}


# Task 1.1

# In[9]:


# Uniform Cost Search - Searches the nodes with the lowest cost first.

def ucs(start, goal, graph):

    # Keep track of the history of visited states and the history of frontier content
    frontier_history = []
    node_history = []

    frontier = PriorityQueue()
    node_start = Node(start, None, None, 0)
    frontier.put((0, node_start))  # Using a priority queue with path cost as the priority
    explored = {}  # Using a dictionary to keep track of the visited state (lowest cost found for each state)
    
    while not frontier.empty():

        # Add the current frontier to the history
        frontier_history.append([extract_path(node) for (cost, node) in frontier.queue])

        _, node = frontier.get()  # Get the node with the lowest path cost
        node_history.append(node.state)

        if node.state == goal:
            return node, node_history, frontier_history # Return the goal node if found
        
        # If the state is not explored or has a lower cost in the explored set, add it to the explored set
        if node.state not in explored or node.path_cost < explored[node.state].path_cost:
            explored[node.state] = node  # Update the explored set with the node

            for child_state, action, step_cost in graph.successors(node.state):
                child_node = Node(child_state, node, action, node.path_cost + step_cost)
                # If the child state is not explored or has a lower cost in the explored set, add it to the frontier
                if child_state not in explored or child_node.path_cost < explored[child_state].path_cost:
                    frontier.put((child_node.path_cost, child_node))
    
    # No solution found
    print("Couldn't find the goal state!")

    return None, node_history, frontier_history


# In[10]:


ucs_sol, ucs_node_history, ucs_frontier_history = ucs('Ipswich', 'Ely', instance)
extract_path(ucs_sol)


# In[11]:


# UCS explored the following states in the the presented order
for i, n in enumerate(ucs_node_history):
    if i < (len(ucs_node_history)-1):
        print(f'{n}', end = " -> ")
    else:
        print(f'{n}')


# In[12]:


# Every step in the fringe
for i, fringe in enumerate(ucs_frontier_history):
    print(f'* At time step {i}, the fringe contained: {fringe}\n')


# Task 1.2

# In[13]:


# Greedy Search (GS) - find the path from the start state to the goal state in a graph, using a provided heuristic function.

def greedySearch(start, goal, graph, heuristic):
    
    # Keep track of the history of visited states and the history of frontier content
    frontier_history = []
    node_history = []
    
    frontier = PriorityQueue()
    frontier.put((heuristic[start], Node(start, None, None, 0))) 
    explored = {}  # Using a dictionary to keep track of the lowest cost found for each state

    while not frontier.empty():

        # Add the current frontier to the history
        frontier_history.append([extract_path(node) for (cost, node) in frontier.queue])

        _, node = frontier.get()  # Get the node with the lowest path cost
        node_history.append(node.state)

        if node.state == goal:
            return node, node_history, frontier_history # Return the goal node if found
        
        # If the state is not explored, add it to the explored set
        if node.state not in explored:
            explored[node.state] = heuristic[node.state]  # Update the explored set with the node

            for child_state, action, step_cost in graph.successors(node.state):
                child_node = Node(child_state, node, action, node.path_cost + step_cost)
                # If the child state is not explored, add it to the frontier
                if child_state not in explored:
                    h = heuristic[child_state]
                    frontier.put((h, child_node))
    
    # No solution found
    print("Couldn't find the goal state!")

    return None, node_history, frontier_history


# In[14]:


greedySearch_sol, greedySearch_node_history, greedySearch_frontier_history = greedySearch('Ipswich', 'Ely', instance, heuristic_structure_value)
extract_path(greedySearch_sol)


# In[15]:


# GS explored the following states in the the presented order
for i, n in enumerate(greedySearch_node_history):
    if i < (len(greedySearch_node_history)-1):
        print(f'{n}', end = " -> ")
    else:
        print(f'{n}')


# In[16]:


# Every step in the fringe
for i, fringe in enumerate(greedySearch_frontier_history):
    print(f'* At time step {i}, the fringe contained: {fringe}\n')


# Task 1.3

# In[17]:


'''
A* Search - to find the path from the start state to the goal state in a graph, using node cost and 
a provided heuristic function.
'''

def aStarSearch(start, goal, graph, heuristic):

    # Keep track of the history of visited states and the history of frontier content
    frontier_history = []
    node_history = []
    
    frontier = PriorityQueue()
    start_node = Node(start, None, None, 0)
    frontier.put((heuristic[start], start_node)) 
    explored = {}  # Using a dictionary to keep track of the lowest cost found for each state

    while not frontier.empty():

        # Add the current frontier to the history
        frontier_history.append([extract_path(node) for (cost, node) in frontier.queue])

        _, node = frontier.get()  # Get the node with the lowest path cost
        node_history.append(node.state)

        if node.state == goal:
            return node, node_history, frontier_history # Return the goal node if found
        
        # If the state is not explored, add it to the explored set
        if node.state not in explored:
            explored[node.state] = node.path_cost  # Update the explored set with the node

            for child_state, action, step_cost in graph.successors(node.state):
                child_node = Node(child_state, node, action, node.path_cost + step_cost)
                # If the child state is not explored or has a lower cost in the explored set, add it to the frontier
                if child_state not in explored or child_node.path_cost < explored[child_state]:
                    h = heuristic[child_state]
                    frontier.put((child_node.path_cost + h, child_node))
    
    # No solution found
    print("Couldn't find the goal state!")

    return None, node_history, frontier_history


# In[18]:


aStarSearch_sol, aStarSearch_node_history, aStarSearch_frontier_history = aStarSearch('Ipswich', 'Ely', instance, heuristic_structure_value)
extract_path(aStarSearch_sol)


# In[19]:


# A* explored the following states in the the presented order
for i, n in enumerate(aStarSearch_node_history):
    if i < (len(aStarSearch_node_history)-1):
        print(f'{n}', end = " -> ")
    else:
        print(f'{n}')


# In[20]:


# Every step in the fringe
for i, fringe in enumerate(aStarSearch_frontier_history):
    print(f'* At time step {i}, the fringe contained: {fringe}\n')


# In[ ]:





# Task 2 - Bayesian Networks

# In[21]:


# Create the Bayesian network structure
travel_BN = BayesianNetwork([
    ("Age", "Education"), 
    ("Sex", "Education"), 
    ("Education", "Residence"), 
    ("Education", "Occupation"),
    ("Residence", "Travel"),
    ("Occupation", "Travel")
])


# In[22]:


# Visualise the Expert-based network

# Create a new graph
G = nx.DiGraph()

# Add the nodes and edges to the graph
G.add_nodes_from(travel_BN.nodes())
G.add_edges_from(travel_BN.edges())

# Draw the graph
nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray')

# Display the graph
plt.show()


# In[23]:


# Load the Travel_data dataset
travel_data = pd.read_csv('Travel_data.csv')


# In[24]:


# Hill-Climbing and K2Score used together to learn the structure of the network
scoring_method = K2Score(data = travel_data)
est = HillClimbSearch(data = travel_data)
estimated_model = est.estimate(scoring_method=scoring_method, max_indegree=3, max_iter=1000)


# In[25]:


# Funtion to evaluate the learned model structures.
def find_f1_score(estimated_model, true_model):
    nodes = estimated_model.nodes()
    est_adj = nx.to_numpy_array(estimated_model.to_undirected(), nodelist=nodes, weight=None)
    true_adj = nx.to_numpy_array(true_model.to_undirected(), nodelist=nodes, weight=None)
    f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score:", round(f1, 3))

find_f1_score(estimated_model, travel_BN)


# In[26]:


# Visualise the learned network

# Create a new graph
G = nx.DiGraph()

# Add the nodes and edges to the graph
G.add_nodes_from(estimated_model.nodes())
G.add_edges_from(estimated_model.edges())

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightgreen', edge_color='gray')

# Display the graph
plt.show()


# In[27]:


# Script learns the parameters of a Bayesian network from data
# Converts the learned network to a BayesianNetwork class object
estimated_model = BayesianNetwork(estimated_model)

# Learn the parameters of the network from the data
estimated_model.fit(travel_data)

# Print the learned parameters
for cpd in estimated_model.get_cpds():
    print(cpd)


# In[28]:


# Difference between age groups and usage of travel modes.
infer = VariableElimination(estimated_model)
q_1 = infer.query(variables = ['Age', 'Travel'],
                evidence = None, # No evidence varibales
                elimination_order = "MinFill", # Can also use "greedy" for example
                show_progress = False)
print("Inference for Age and Travel:")
print(q_1)


# In[29]:


# Difference between males and females of travel modes
infer = VariableElimination(estimated_model)
q_2 = infer.query(variables = ['Sex', 'Travel'],
                evidence = None, # No evidence varibales
                elimination_order = "MinFill",
                show_progress = False)
print("Inference for Sex and Travel:")
print(q_2)


# In[30]:


# Difference between males of travel mode
infer = VariableElimination(estimated_model)
q_males = infer.query(variables = ['Travel'],
                evidence = {'Sex': 'M'}, 
                elimination_order = "MinFill",
                show_progress = False)
print("Inference for Males and Travel:")
print(q_males)


# In[31]:


# Difference between females of travel mode
infer = VariableElimination(estimated_model)
q_females = infer.query(variables = ['Travel'],
                evidence = {'Sex': 'F'}, 
                elimination_order = "MinFill",
                show_progress = False)
print("Inference for Females and Travel:")
print(q_females)


# In[32]:


# Learn parameters of Bayesian Network
travel_BN.fit(travel_data)


# In[33]:


# Comparing the results from the expert-based and learned models
def compare_inferences(model1, model2, evidence, variables):
    infer1 = VariableElimination(model1)
    infer2 = VariableElimination(model2)
    result1 = infer1.query(variables=variables, evidence=evidence)
    result2 = infer2.query(variables=variables, evidence=evidence)
    print(f"Results for evidence {evidence}:")
    print("Expert-based model:")
    print(result1)
    print("Learned model:")
    print(result2)
    
# Check the comparisons of given variables
compare_inferences(travel_BN, estimated_model, evidence={'Sex': 'M'}, variables=['Travel'])
compare_inferences(travel_BN, estimated_model, evidence={'Age': 'young'}, variables=['Travel'])


# In[34]:


# Split the data into train and test sets
train_data, test_data = train_test_split(travel_data, test_size=0.2, random_state=42)


# In[35]:


# Fit models on the training data
travel_BN.fit(train_data)
estimated_model.fit(train_data)


# In[36]:


# Evaluate models on the data using BIC score
expert_score = BicScore(test_data).score(travel_BN)
learned_score = BicScore(test_data).score(estimated_model)

print(f"Expert-based model BIC score: {round(expert_score, 2)}")
print(f"Learned model BIC score: {round(learned_score, 2)}")


# In[37]:


# Determine which model is more accurate
if learned_score < expert_score:
    print("The learned model predicts the travel data more accurately.")
else:
    print("The expert-based model predicts the travel data more accurately.")


# In[ ]:





# Task 3 – Machine Learning

# In[38]:


# Load the datasets
train_data = pd.read_csv('twitter_train.csv')
test_data = pd.read_csv('twitter_test.csv')


# In[39]:


# Display the first few rows of the datasets from twitter_train.csv
print(train_data.head())


# In[40]:


# Display the first few rows of the datasets from twitter_test.csv
print(test_data.head())


# In[51]:


# Function to clean the text
def clean_data(text):
    text = "".join([char.lower() for char in text if char not in string.punctuation]) # Remove punctuation
    tokens = re.split(r'\W+', text) # Tokenize
    ps = nltk.PorterStemmer() # Remove stem
    
    # Remove stopwords
    stopword_list = stopwords.words('english') 
    text = [ps.stem(word) for word in tokens if word not in stopword_list]
    return text

# Apply it to the whole dataset
train_data['clean text'] = train_data['text'].apply(clean_text)
test_data['clean text'] = test_data['text'].apply(clean_text)

# Display the resulting datasets
print("twitter_train.csv:")
print(train_data.head())

print("\n\ntwitter_test.csv:")
print(test_data.head())


# In[52]:


# Initialise CountVectoriser based on the text cleaning function
count_vect = CountVectorizer(analyzer=clean_data, max_features=3000)

# Create the document-term matrix based on counts
X_train_counts = count_vect.fit_transform(train_data['clean text'])
X_test_counts = count_vect.fit_transform(test_data['clean text'])


# In[43]:


# Labels
y_train = train_data['sentiment']
y_test = test_data['sentiment']

# Initialise Naive Bayes
nb = MultinomialNB()

# Explores the parameters
parameters = {'alpha': [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

# Grid search
np.random.seed(0) # shows same results everytime it runs

grid_search = GridSearchCV(nb, parameters, cv=5, scoring = 'f1_micro', n_jobs=-1)  # Evaluate F1-score
grid_search_fit = grid_search.fit(X_train_counts, y_train)
pd.DataFrame(grid_search_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# In[44]:


# Train Naive Bayes model with the best parameters from grid search
best_nb = grid_search_fit.best_estimator_
best_nb.fit(X_train_counts, y_train)


# In[65]:


# Make predictions on the training set
train_pred = best_nb.predict(X_train_counts)
print('Naive Bayes - Overall accuracy on training set:', round(accuracy_score(y_train, train_pred), 3))


# In[62]:


# Evaluate the model on the test set
test_pred = best_nb.predict(X_test_counts)
precision_nb, recall_nb, fscore_nb, _ = score(y_test, test_pred, pos_label='negative', average='binary')
accuracy_nb = accuracy_score(y_test, test_pred)
print('Naive Bayes - Accuracy: {} / Precision: {} / Recall: {} / F1-score: {}'.format(
    round(accuracy_nb, 3),
    round(precision_nb, 3),
    round(recall_nb, 3),
    round(fscore_nb, 3)))


# In[69]:


# Display the calculated graph
ConfusionMatrixDisplay.from_predictions(y_test, test_pred)
plt.show()


# In[66]:


# Train and evaluate Logistic Regression for comparison
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_counts, y_train)
y_pred_lr = lr.predict(X_test_counts)

precision_lr, recall_lr, fscore_lr, _ = score(y_test, y_pred_lr, pos_label='negative', average='binary')
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print('Logistic Regression - Accuracy: {} / Precision: {} / Recall: {} / F1-score: {}'.format(
    round(accuracy_lr, 3),
    round(precision_lr, 3),
    round(recall_lr, 3),
    round(fscore_lr, 3)))


# In[70]:


# Display the calculated graph
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.show()


# In[67]:


# Compare both models
print("Model Comparison:")
print(f"Naive Bayes - Accuracy: {round(accuracy_nb, 3)}, Precision: {round(precision_nb, 3)}, Recall: {round(recall_nb, 3)}, F1-score: {round(fscore_nb,3)}")
print(f"Logistic Regression - Accuracy: {round(accuracy_lr, 3)}, Precision: {round(precision_lr, 3)}, Recall: {round(recall_lr, 3)}, F1-score: {round(fscore_lr, 3)}")


# In[68]:


# Compare both models using accuracy
if accuracy_lr > accuracy_nb:
    print("Logistic Regression is more accurate than Naive Bayes.")
elif accuracy_lr < accuracy_nb:
    print("Naive Bayes is more accurate than Logistic Regression.")
else:
    print("Both models have the same accuracy.")


# In[ ]:




