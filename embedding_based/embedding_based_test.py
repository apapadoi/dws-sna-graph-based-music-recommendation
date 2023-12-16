import json
from concurrent.futures import ThreadPoolExecutor
import time
import sys
import os
import gc

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import seaborn as sns

# TODO check if it is automatically executed in CUDA when available
if len(sys.argv) == 1:
    raise RuntimeError("Missing command line argument")
elif sys.argv[1] == 'challenge':
    file_path = './challenge_set.json'

    with open(file_path, 'r') as file:
        data = json.load(file)

        del data['date']
        del data['version']
        del data['name']
        del data['description']

        playlists = data['playlists']

        print(f'Total playlists: {len(playlists)}')

        non_empty_playlists = []
        playlist_lengths = []

        for current_playlist in playlists:
            if len(current_playlist['tracks']) >= 3:
                non_empty_playlists.append(current_playlist)

        print(f'Non-empty playlists: {len(non_empty_playlists)}')

        # TODO tracks need preprocessing, e.g. No Strings Attached in album name field or (possibly) other fields
        
        train_playlists = []
        test_playlists = []

        for current_playlist in non_empty_playlists:
            playlist_lengths.append(len(current_playlist['tracks']))
            train_playlists.append(current_playlist['tracks'][:-1])
            test_playlists.append([current_playlist['tracks'][-1]])
            
            assert len(current_playlist['tracks']) == len(train_playlists[-1]) + len(test_playlists[-1]) and len(train_playlists[-1]) == len(current_playlist['tracks']) - 1
    
        # train_playlists = train_playlists[:1000] # TODO remove this

        training_multi_graph = nx.MultiGraph()

        counter = 0
        for current_train_playlist in train_playlists:
            for current_first_track_index, current_first_track in enumerate(current_train_playlist):
                if current_first_track_index == len(current_train_playlist) - 1:
                    break

                current_second_track = current_train_playlist[current_first_track_index + 1]
                if f"{current_first_track['track_name']}_{current_first_track['artist_name']}_{current_first_track['album_name']}" == 'Naive_The Kooks_Inside In / Inside Out' and f"{current_second_track['track_name']}_{current_second_track['artist_name']}_{current_second_track['album_name']}" == 'Mr. Brightside_The Killers_Inside In / Inside Out':
                    counter += 1
                training_multi_graph.add_edge(f"{current_first_track['track_name']}_{current_first_track['artist_name']}_{current_first_track['album_name']}", f"{current_second_track['track_name']}_{current_second_track['artist_name']}_{current_second_track['album_name']}")


        print(f'counter: {counter}')
        print(f'Number of nodes: {training_multi_graph.number_of_nodes()}')
        print(f'Number of edges: {training_multi_graph.number_of_edges()}')


        training_undirected_graph = nx.Graph()

        for u, v, data in training_multi_graph.edges(data=True):
            if training_undirected_graph.has_edge(u, v):
                training_undirected_graph[u][v]['weight'] += 1
            else:
                training_undirected_graph.add_edge(u, v, weight=1)
else:
    TRAIN_PERCENTAGE_PER_FILE = 0.7
    folder_path = './spotify_million_playlist_dataset/data'

    def extract_x_y(file_name):
        parts = file_name.split('.')[2]
        x_y = parts.split('-')
        return int(x_y[0]), int(x_y[1])

    json_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.json')], key=extract_x_y)
    total_playlists = 0
    total_non_empty_playlists = 0
    test_playlists = []
    training_undirected_graph = nx.Graph()
    playlist_lengths = []
    
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        if extract_x_y(file_name)[0] % 10000 == 0:
            print(file_name)
            gc.collect()

        if extract_x_y(file_name)[0] == 10000:
            break
            
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

            del data['info']

            playlists = data['playlists']

            total_playlists += len(playlists)

            for index, current_playlist in enumerate(playlists):
                if len(current_playlist['tracks']) >= 3:
                    playlist_lengths.append(len(current_playlist['tracks']))

                    if index < len(playlists) * TRAIN_PERCENTAGE_PER_FILE:
                        current_train_playlist = current_playlist['tracks']

                        for current_first_track_index, current_first_track in enumerate(current_train_playlist):
                            if current_first_track_index == len(current_train_playlist) - 1:
                                break

                            current_second_track = current_train_playlist[current_first_track_index + 1]

                            u = f"{current_first_track['track_name']}_{current_first_track['artist_name']}_{current_first_track['album_name']}"
                            v = f"{current_second_track['track_name']}_{current_second_track['artist_name']}_{current_second_track['album_name']}"

                            if training_undirected_graph.has_edge(u, v):
                                training_undirected_graph[u][v]['weight'] += 1
                            else:
                                training_undirected_graph.add_edge(u, v, weight=1)
                    else:
                        test_playlists.append([f"{track['track_name']}_{track['artist_name']}_{track['album_name']}" for track in current_playlist['tracks']])

                    total_non_empty_playlists += 1
            

    print(f'Total playlists: {total_playlists}')
    print(f'Non-empty playlists: {total_non_empty_playlists}')


print(f'Number of nodes (undirected graph): {training_undirected_graph.number_of_nodes()}')
print(f'Number of edges (undirected graph): {training_undirected_graph.number_of_edges()}')


degrees = sorted([degree for node, degree in training_undirected_graph.degree()], reverse=True)

plt.figure(figsize=(20, 10))
sns.histplot(degrees)
plt.xscale('log')
plt.yscale('log')  
plt.title('Log-log plot of degree distribution (Spotify Million Playlist Dataset - training only)', fontsize=16)
plt.xlabel('Log-degree', fontsize=16)
plt.ylabel('Log-frequency', fontsize=16)
plt.grid(True)
plt.savefig("training_undirected_degree_distribution_plot.png")

plt.figure(figsize=(20, 10))
sns.histplot(playlist_lengths)
plt.yscale('log')  
plt.title('Log plot of playlist length (based on #songs) distribution (Spotify Million Playlist Dataset - both on training and testing)', fontsize=16)
plt.xlabel('Playlist length (based on #songs)', fontsize=16)
plt.ylabel('Log-frequency', fontsize=16)
plt.grid(True)
plt.savefig("training_undirected_playlist_length_distribution_plot.png")

EMBEDDING_SIZE = 8
NEGATIVE_POSITIVE_SAMPLE_RATIO = 1
RANDOM_STATE = 42
LEARNING_RATE = 0.01
NUM_EPOCHS = 1
NUM_WORKERS = 12
NEGATIVE_CANDIDATE_SONGS = 10
K = 5

torch.manual_seed(RANDOM_STATE)

indexing_per_node = {}
non_neighbors_per_node = {}
neighbors_per_node = {}

rng = np.random.default_rng(seed=RANDOM_STATE)

def process_node(node, index):
    # if index % 10000 == 0:
    # print(index, node)
    neighbors_result = list(training_undirected_graph.neighbors(node))
    non_neighbors_result = []
    num_iterations = 0

    for non_neighbor in nx.non_neighbors(training_undirected_graph, node):
        if num_iterations == len(neighbors_result) * NEGATIVE_POSITIVE_SAMPLE_RATIO:
            break

        non_neighbors_result.append(non_neighbor)

        num_iterations += 1

    assert len(non_neighbors_result) == len(neighbors_result) * NEGATIVE_POSITIVE_SAMPLE_RATIO
    return node, index, non_neighbors_result, neighbors_result


with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(process_node, node, index): (node, index) for index, node in enumerate(list(training_undirected_graph.nodes()))}
    for future in futures:
        node, indexing, non_neighbors, neighbors = future.result()
        indexing_per_node[node] = indexing
        non_neighbors_per_node[node] = non_neighbors
        neighbors_per_node[node] = neighbors


print("Indexing per node:", len(list(indexing_per_node.keys())))
print("Non-neighbors per node:", len(list(non_neighbors_per_node.keys())))
print("Neighbors per node:", len(list(neighbors_per_node.keys())))


embeddings = torch.randn(len(list(training_undirected_graph.nodes())), EMBEDDING_SIZE, requires_grad=True)
print('embeddings before')
print(embeddings.shape)
print(embeddings[0])

start_time = time.time()

def embedding_based_loss_function(embeddings):
    loss = 0
    for node in list(training_undirected_graph.nodes()): # TODO try expressing this for also as tensor multiplication
        current_internal_sum = 0.0

        neighbor_tensor_list = []
        weight_tensor_list = []
        for neighbor in neighbors_per_node[node]:
            neighbor_tensor_list.append(embeddings[indexing_per_node[neighbor]])
            weight_tensor_list.append(torch.tensor(training_undirected_graph.get_edge_data(node, neighbor)['weight']))

        neighbor_tensor_list_stacked = torch.stack(neighbor_tensor_list)
        weight_tensor_list_stacked = torch.stack(weight_tensor_list)

        expanded_node_tensor = embeddings[indexing_per_node[node]].expand_as(neighbor_tensor_list_stacked)
        lambda_i_j = torch.exp(-torch.norm(expanded_node_tensor - neighbor_tensor_list_stacked, dim=1, p=2)) # TODO integrate bias (edit distance between album titles of corresponding song pair [just an idea to include some NLP too] - normalized to [0,1]? - use of duration?)
        current_internal_sum = torch.sum(lambda_i_j - weight_tensor_list_stacked * torch.log10(lambda_i_j))

        non_neighbor_tensor_list = []
        
        for non_neighbor in non_neighbors_per_node[node]:
            non_neighbor_tensor_list.append(embeddings[indexing_per_node[non_neighbor]])
        
        non_neighbor_tensor_list_stacked = torch.stack(non_neighbor_tensor_list)
        lambda_i_j = torch.exp(-torch.norm(expanded_node_tensor - non_neighbor_tensor_list_stacked, dim=1, p=2)) # TODO integrate bias (edit distance between album titles of corresponding song pair [just an idea to include some NLP too] - normalized to [0,1]? - use of duration?)
        poisson_tensor_list = torch.poisson(lambda_i_j)

        current_internal_sum += torch.sum(lambda_i_j - poisson_tensor_list * torch.log10(lambda_i_j))

        loss += current_internal_sum

    print(f'loss: {loss}')
    
    return loss


optimizer = optim.Adam([embeddings], lr=LEARNING_RATE)


for epoch in range(NUM_EPOCHS):
    print(f'epoch: {epoch}/{NUM_EPOCHS}')
    optimizer.zero_grad()
    loss = embedding_based_loss_function(embeddings)
    loss.backward()
    optimizer.step()

end_time = time.time()

execution_time = end_time - start_time
print(f"Total Execution Time: {execution_time:.4f} seconds")


optimized_embeddings = embeddings.detach().numpy()

print('embeddings after')
print(optimized_embeddings.shape)
print(optimized_embeddings[0])

# Evaluation
# TODO evaluation can be done in parallel with ThreadPoolExecutor
training_graph_nodes = list(training_undirected_graph.nodes())
testing_ranks_per_transition = []
for index, current_test_playlist in enumerate(test_playlists):
    print(f'test playlist {index}/{len(test_playlists)}')
    current_seen_tracks = []
    for current_first_track_index, current_first_track in enumerate(current_test_playlist):
        if current_first_track_index == len(current_test_playlist) - 1:
            break

        current_seen_tracks.append(current_first_track)

        current_second_track = current_test_playlist[current_first_track_index + 1]

        current_candidate_song_list = [current_second_track]

        for i in range(NEGATIVE_CANDIDATE_SONGS):
            random_index = rng.integers(0, len(training_graph_nodes))
            current_negative_candidate_track = training_graph_nodes[random_index]
            while current_negative_candidate_track in current_seen_tracks:
                random_index = rng.integers(0, len(training_graph_nodes))
                current_negative_candidate_track = training_graph_nodes[random_index]
            
            current_candidate_song_list.append(current_negative_candidate_track)


        if current_first_track not in training_graph_nodes:
            optimized_embedding_to_use_for_first_track = rng.standard_normal(EMBEDDING_SIZE)
        else:
            optimized_embedding_to_use_for_first_track = optimized_embeddings[indexing_per_node[current_first_track]]

        embeddings_for_candidate_song_list = []
        for track in current_candidate_song_list:
            if track in training_graph_nodes:
                embeddings_for_candidate_song_list.append((track, optimized_embeddings[indexing_per_node[track]]))
            else:
                embeddings_for_candidate_song_list.append((track, rng.standard_normal(EMBEDDING_SIZE)))


        distances_from_first_track = [np.linalg.norm(array[1] - optimized_embedding_to_use_for_first_track) for array in embeddings_for_candidate_song_list]
        
        indexed_distances = list(enumerate(distances_from_first_track))

        sorted_distances = sorted(indexed_distances, key=lambda x: x[1])

        sorted_indices = [index for index, _ in sorted_distances]

        sorted_candidate_song_list = [embeddings_for_candidate_song_list[index] for index in sorted_indices]

        correct_song_rank = -1
        for current_candidate_song_index, song in enumerate(sorted_candidate_song_list):
            if song[0] == current_second_track:
                correct_song_rank = current_candidate_song_index + 1
                break

        assert correct_song_rank > 0
        testing_ranks_per_transition.append(correct_song_rank)

print('test')

reciprocal_ranks = [1 / rank for rank in testing_ranks_per_transition if rank != 0]
mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

print("Mean Reciprocal Rank (MRR):", mrr)

relevant_items = sum(1 for rank in testing_ranks_per_transition if rank <= K)
total_transitions = len(testing_ranks_per_transition)

recall_at_k = relevant_items / total_transitions if total_transitions != 0 else 0.0

print(f"Recall@{K}: {recall_at_k}")
# TODO save embeddings in order to have them pre-computed, visualize some of them and implement baselines for comparisons, effect of learning rate?