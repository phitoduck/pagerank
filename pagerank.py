"""


The Page Rank Algorithm.

Eric Riddoch
Mar 5, 2019


"""

import numpy as np
from scipy import linalg as la
import csv
import networkx as nx

class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    """

    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        
        # store attributes
        A = np.copy(A)
        self.n = A.shape[0] # matrix is assumed to be square

        # default labels are [1, 2, ..., n - 1]
        if labels is not None:
            self.labels = labels
            if len(labels) != self.n:
                raise ValueError("Wrong labels size!")
        else:
            self.labels = np.arange(self.n)
            
        # replace columns of zeroes with columns of ones
        for j in range(self.n):
            if np.all(A[:, j] == 0):  # check that j'th column is all zeroes
                A[:, j] = 1
                
        # divide each entry by columns sum
        self.A_hat = A / A.sum(axis=0)
        
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        
        # solve for page rank vector using matrix solver
        matrix = np.eye(self.n) - epsilon*self.A_hat
        vector = ((1 - epsilon) / self.n) * np.ones(self.n)
        
        self.page_rank_vector = la.solve(matrix, vector)
        
        # return dictionary of the form { "node_label" (string) : page_rank (float) }
        return { label : self.page_rank_vector[label_index] for label_index, label in enumerate(self.labels) }

    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        
        # find B
        E = np.ones_like(self.A_hat)
        B = epsilon*self.A_hat + ((1 - epsilon) / self.n)*E
        
        # find the eigenvectors of B
        eig_vects = la.eig(B)
        eig_vector = self.page_rank_vector = eig_vects[1][:, 0]
        self.page_rank_vector = eig_vector / la.norm(eig_vector, ord=1)
        
        # return dictionary of the form { "node_label" (string) : page_rank (float) }
        return { label : self.page_rank_vector[label_index] for label_index, label in enumerate(self.labels) }

    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        
        p0 = np.full(self.n, 1 / self.n).reshape(self.n)  # initial state
        ones = np.ones(self.n)
        
        for i in range(maxiter):
            p1 = epsilon*self.A_hat @ p0 + (1-epsilon)/self.n*ones            # compute next iteration
            if la.norm(p1 - p0, ord=1) < tol:        # check if we found steady state
                break
            p0 = p1.reshape(self.n)   
            
        return { label : p0[label_index] for label_index, label in enumerate(self.labels) }

def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    
    # return list of keys sorted by values in descending order
    return sorted(d, key=d.get, reverse=True)

def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    
    # load stanford files into dictionary
    graph = dict()
    
    # 1. get a dictionary of the form { node_id : column_index }
    labels = set()
    with open(filename, 'r') as file: # loop through each line
        for line in file:
            node = line.strip().split("/")
            labels.update(node)
    labels = sorted(list(labels))
    page_ids = dict(zip(labels, range(len(labels)))) # zip labels and indices into tuples
    
    # 2. build the matrix
    n = len(page_ids.keys())
    A = np.zeros((n, n))              # declare an n x n matrix of zeroes
    with open(filename, 'r') as file: # loop through each line
        for line in file:
            node = line.strip().split("/")
            label = node[0]
            col_index = page_ids[label]
            for row_label in node[1:]:
                row_index = page_ids[row_label]
                A[row_index, col_index] = 1
        
    # 3. find page rank
    graph = DiGraph(A, labels)
    page_rank_vector = graph.itersolve()
    ranks = get_ranks(page_rank_vector)
    
    return ranks

def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    
    # 1. read in data
    data = np.genfromtxt(filename, dtype=object, delimiter=",")
    
    # 2. identify teams
    teams = list(set(data[1:, 0]) | set(data[1:, 1]))
    
    # 3. associate each team with a number
    indices = dict(zip(teams, range(len(teams))))
    
    # 4. build adjacency matrix where the i, jth entry represents
    # the number of times team i beat team j
    n = len(teams)
    matrix = np.zeros((n, n))
    for row in data[1:]:
        winner, loser = row
        matrix[indices[winner], indices[loser]] += 1
    
    # 5. find page rank
    graph = DiGraph(matrix, teams)
    page_rank_vector = graph.itersolve(epsilon=epsilon)
    ranks = get_ranks(page_rank_vector)
    
    return ranks

def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    
    # graph
    graph = nx.DiGraph()
    
    def update_edge(actor1, actor2):
        if graph.has_edge(actor1, actor2):
            graph[actor1][actor2]['weight'] += 1
        else:
            graph.add_edge(actor1, actor2, weight=1)
    
    # read file
    with open(filename, 'r') as file:
        for line in file:                           # read each line
            actors = line.strip().split('/')[1:]    # get actors
            for i in range(len(actors)):            # add actor edges to graph
                for j in range(i):
                    update_edge(actors[i], actors[j])
                    
    # compute page rank of each actor
    page_rank = nx.pagerank(graph, alpha=epsilon)
    
    return get_ranks(page_rank)
            