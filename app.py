import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def generate_nodes(n):
    nodes = []
    for i in range(n):
        node = {'teamwork': random.randint(0, 10),
                'communication': random.randint(0, 10),
                'problem_solving': random.randint(0, 10),
                'adaptability': random.randint(0, 10)}
        nodes.append(node)
    return nodes

st.title("Alpha Prototype: Network Analysis")
st.write("This prototype uses random data for the purpose of testing and demonstration. The treshold for establishing connections is set at default as 0.8 for the purpose of this prototype but in reality the user can set the treshold")

# Step 1: Input number of team members and new candidates
st.header("Step 1: Input number of team members and new candidates")

num_team_members = st.number_input("Number of team members", min_value=1, max_value=20, value=10, step=1)
num_candidates = st.number_input("Number of new candidates", min_value=1, max_value=20, value=4, step=1)

if st.button("Generate Random Data for Team Members and Candidates and Perform Network Analysis to Find the Best Fit"):
    nodes = generate_nodes(num_team_members)
    new_nodes = generate_nodes(num_candidates)

    # Show team members and candidates data in tables
    st.write("Random data for the team members:")
    st.write(pd.DataFrame(nodes))
    st.write("Random data for the new candidates:")
    st.write(pd.DataFrame(new_nodes))

    # Step 2: Display graph before and after introducing new nodes
    st.header("Step 2: Network Analysis")

    # Calculating the cosine similarity matrix
    similarity_matrix = np.zeros((num_team_members, num_team_members))
    for i in range(num_team_members):
        for j in range(num_team_members):
            a = [nodes[i]['teamwork'], nodes[i]['communication'], nodes[i]['problem_solving'], nodes[i]['adaptability']]
            b = [nodes[j]['teamwork'], nodes[j]['communication'], nodes[j]['problem_solving'], nodes[j]['adaptability']]
            similarity_matrix[i][j] = cosine_similarity(a, b)

    # Creating a networkx graph and adding edges based on similarity matrix
    G = nx.Graph()
    for i in range(num_team_members):
        for j in range(num_team_members):
            if i != j and similarity_matrix[i][j] > 0.8:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    # Plotting the graph before introducing new nodes
    nx.draw(G, with_labels=True)
    plt.title("Network before introducing new nodes")
    st.pyplot(plt.gcf())
    plt.clf()

    # Add new candidates to the graph
    for i in range(num_candidates):
        G.add_node(num_team_members + i, teamwork=new_nodes[i]['teamwork'], communication=new_nodes[i]['communication'],
                   problem_solving=new_nodes[i]['problem_solving'], adaptability=new_nodes[i]['adaptability'])

    # Connect the new nodes to the original network
    for i in range(num_candidates):
        for j in range(len(nodes)):
            a = [new_nodes[i]['teamwork'], new_nodes[i]['communication'], new_nodes[i]['problem_solving'], new_nodes[i]['adaptability']]
            b = [nodes[j]['teamwork'], nodes[j]['communication'], nodes[j]['problem_solving'], nodes[j]['adaptability']]
            similarity = cosine_similarity(a, b)
            if i != j and similarity > 0.8:
                G.add_edge(num_team_members + i, j, weight=similarity)

    # Plotting the graph after connecting the new nodes
    colors = ["blue" if i >= num_team_members else "red" for i in G.nodes()]
    nx.draw(G, with_labels=True, node_color=colors)
    plt.title("Network after connecting the new nodes")
    st.pyplot(plt.gcf())
    plt.clf()


    st.subheader("Step 3: Analyzing the network")

    # Calculating the eigen vector centrality
    eigen_centrality = nx.eigenvector_centrality_numpy(G)

    # Finding the new node with the highest eigen vector centrality
    max_centrality = 0
    max_node = None
    for node in G.nodes():
        if node >= num_team_members:
            centrality = eigen_centrality[node]
            if centrality > max_centrality:
                max_centrality = centrality
                max_node = node

    # Displaying the node with the highest eigen vector centrality
    st.write("Node with highest eigen vector centrality:", max_node)

    # Calculating the quality of connections to the old nodes in the network
    total_similarity = 0
    num_connections = 0
    for node in G.nodes():
        if node < num_team_members:
            similarity = G.get_edge_data(max_node, node, default={'weight': 0})['weight']
            if similarity > 0:
                total_similarity += similarity
                num_connections += 1

    if num_connections == 0:
        avg_similarity = 0
    else:
        avg_similarity = total_similarity / num_connections

    # Displaying the quality of connections to the old nodes in the network
    st.write("Quality of connections to old nodes in the network:", avg_similarity)

    st.subheader("Step 4: Calculating Confidence")

    # Calculating the quality of connections to the old nodes in the network
    total_similarity = 0
    num_connections = 0
    for node in G.nodes():
        if node < num_team_members:
            similarity = G.get_edge_data(max_node, node, default={'weight': 0})['weight']
            if similarity > 0:
                total_similarity += similarity
                num_connections += 1

    if num_connections == 0:
        avg_similarity = 0
    else:
        avg_similarity = total_similarity / num_connections

    confidence = avg_similarity * 100

    # Displaying the confidence of the suggestion
    st.write("Confidence of the suggestion :", confidence, "%")

    st.subheader("Step 5: Calculating Pearson Correlation Coefficients")

    # Extracting the attribute values for the best fit candidate
    best_fit_attributes = np.array([new_nodes[max_node - num_team_members]['teamwork'], new_nodes[max_node - num_team_members]['communication'], 
                                    new_nodes[max_node - num_team_members]['problem_solving'], new_nodes[max_node - num_team_members]['adaptability']])

    # Extracting the attribute values for the original team members
    team_member_attributes = np.array([[node['teamwork'], node['communication'], node['problem_solving'], node['adaptability']] for node in nodes])

    # Calculating the Pearson correlation coefficients between the best fit candidate and each team member
    correlation_coefficients = np.array([np.corrcoef(best_fit_attributes, team_member_attributes[i])[0, 1] for i in range(len(team_member_attributes))])

    # Displaying the correlation coefficients
    st.write("Pearson correlation coefficients between the best fit candidate and the original team members:")
    for i in range(len(correlation_coefficients)):
        st.write(f"Team member {i+1}: {correlation_coefficients[i]}")

    st.subheader("Step 6: Quality of Connections of the New Node to Connected Old Nodes")

    # Calculating the eigenvector centrality of nodes
    centrality = nx.eigenvector_centrality(G)

    # Finding the new node with highest eigenvector centrality
    new_node_ids = [i for i in range(num_team_members, num_team_members + num_candidates)]
    new_node_centrality = [centrality[i] for i in new_node_ids]
    best_fitted_node = new_node_ids[np.argmax(new_node_centrality)]

    # Displaying the best fitted node
    st.write(f"The best fitted node in the network is: {best_fitted_node}")

    # Calculating the quality of connections to old nodes
    quality = {}
    for j in range(num_team_members):
        a = [new_nodes[best_fitted_node - num_team_members]['teamwork'], new_nodes[best_fitted_node - num_team_members]['communication'], 
             new_nodes[best_fitted_node - num_team_members]['problem_solving'], new_nodes[best_fitted_node - num_team_members]['adaptability']]
        b = [nodes[j]['teamwork'], nodes[j]['communication'], nodes[j]['problem_solving'], nodes[j]['adaptability']]
        similarity = cosine_similarity(a, b)
        if similarity > 0.8:
            quality[j] = similarity * 100

    # Displaying the quality of connections to old nodes
    st.write("The quality of connections to old nodes are:")
    for key, value in quality.items():
        st.write(f"Node {key}: {value}%")

    import pandas as pd

    st.subheader("Step 7: Summary Table and Graphs")

    # Creating a summary table
    summary_data = {
        'Team Member': list(range(num_team_members)) + [f'Candidate {i - num_team_members + 1}' for i in range(num_team_members, num_team_members + num_candidates)],
        'Teamwork': [node['teamwork'] for node in nodes] + [new_node['teamwork'] for new_node in new_nodes],
        'Communication': [node['communication'] for node in nodes] + [new_node['communication'] for new_node in new_nodes],
        'Problem Solving': [node['problem_solving'] for node in nodes] + [new_node['problem_solving'] for new_node in new_nodes],
        'Adaptability': [node['adaptability'] for node in nodes] + [new_node['adaptability'] for new_node in new_nodes],
        'Eigenvector Centrality': [centrality[node_id] for node_id in range(num_team_members + num_candidates)],
        'Quality of Connections': [quality.get(i, 0) for i in range(num_team_members)] + [0] * num_candidates,
        'Pearson Correlation': list(correlation_coefficients) + [0] * num_candidates
    }

    summary_df = pd.DataFrame(summary_data)
    summary_df['Team Member'] = summary_df['Team Member'].astype(str)


    # Displaying the summary table
    st.write(summary_df)

    # Plotting graphs to visualize outputs
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    # Teamwork and Communication scatter plot
    ax[0, 0].scatter(summary_df['Team Member'], summary_df['Teamwork'], label='Teamwork', color='blue')
    ax[0, 0].scatter(summary_df['Team Member'], summary_df['Communication'], label='Communication', color='red')
    ax[0, 0].set_title('Teamwork and Communication')
    ax[0, 0].legend()

    # Problem Solving and Adaptability scatter plot
    ax[0, 1].scatter(summary_df['Team Member'], summary_df['Problem Solving'], label='Problem Solving', color='green')
    ax[0, 1].scatter(summary_df['Team Member'], summary_df['Adaptability'], label='Adaptability', color='purple')
    ax[0, 1].set_title('Problem Solving and Adaptability')
    ax[0, 1].legend()

    # Eigenvector Centrality bar plot
    ax[1, 0].bar(summary_df['Team Member'], summary_df['Eigenvector Centrality'], color='orange')
    ax[1, 0].set_title('Eigenvector Centrality')

    # Quality of Connections bar plot
    ax[1, 1].bar(summary_df['Team Member'], summary_df['Quality of Connections'], color='brown')
    ax[1, 1].set_title('Quality of Connections')

    # Adjusting the layout and displaying the graphs
    plt.tight_layout()
    st.pyplot(fig)




